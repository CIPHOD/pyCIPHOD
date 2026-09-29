from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression


# ============================================================================
# Results
# ============================================================================


@dataclass
class CIRCANodeResult:
    """
    CIRCA result for one variable.

    raw_score:
        Regression-based hypothesis testing (RHT) score.

    adjusted_score:
        Score after CIRCA descendant adjustment.

    max_anomalous_z:
        Maximum absolute residual z-score in the anomalous regime.

    parents:
        Graph parents used in the regression.

    features:
        Actual regression features, including temporal lags.

    normal_residual_mean:
        Mean residual estimated from the normal regime.

    normal_residual_std:
        Standard deviation of residuals estimated from the normal regime.
    """

    node: str
    raw_score: float
    adjusted_score: float
    max_anomalous_z: float
    parents: Tuple[str, ...]
    features: Tuple[str, ...]
    normal_residual_mean: float
    normal_residual_std: float


# ============================================================================
# CIRCA
# ============================================================================


class CIRCA:
    """
    Causal Inference-based Root Cause Analysis (CIRCA).

    This implementation assumes that the causal graph is already available.
    Therefore, it implements CIRCA's scoring stages:

        1. Regression-based hypothesis testing (RHT)
        2. Descendant adjustment
        3. Root-cause ranking

    The graph is expected to expose:

        graph.get_parents(node)
        graph.get_directed_edges()

    which matches the graph interface already used in pyciphod.

    Parameters
    ----------
    regressor:
        sklearn-compatible regression model. By default:
        LinearRegression().

    parent_max_lag:
        For every graph parent X -> Y, include

            X_t, X_{t-1}, ..., X_{t-parent_max_lag}

        as regression predictors.

        parent_max_lag=0 corresponds to contemporaneous parents only.

    include_self_lag:
        Whether to include past values of Y when modelling Y_t.

        This is particularly relevant to your simulation, where each
        variable has a lag-1 self-dependence.

    self_max_lag:
        If include_self_lag=True, use

            Y_{t-1}, ..., Y_{t-self_max_lag}.

    descendant_adjustment:
        Apply CIRCA's descendant adjustment after RHT.

        This requires the cross-variable graph, after removing self-loops,
        to be a DAG.

    anomaly_threshold:
        Threshold used in descendant adjustment. CIRCA uses the
        three-sigma rule, i.e. 3.

    normal_window:
        Optional number of observations immediately preceding anomaly_start
        to use for regression training. If None, all available pre-anomaly
        observations are used.

    eps:
        Numerical tolerance used when the residual standard deviation is
        approximately zero.
    """

    def __init__(
        self,
        *,
        regressor=None,
        parent_max_lag: int = 0,
        include_self_lag: bool = False,
        self_max_lag: int = 1,
        descendant_adjustment: bool = True,
        anomaly_threshold: float = 3.0,
        normal_window: Optional[int] = None,
        eps: float = 1e-12,
    ):
        if parent_max_lag < 0:
            raise ValueError("parent_max_lag must be non-negative.")

        if self_max_lag < 1:
            raise ValueError("self_max_lag must be at least 1.")

        if anomaly_threshold < 0:
            raise ValueError("anomaly_threshold must be non-negative.")

        if normal_window is not None and normal_window <= 0:
            raise ValueError("normal_window must be positive.")

        self.regressor = (
            LinearRegression()
            if regressor is None
            else regressor
        )

        self.parent_max_lag = int(parent_max_lag)
        self.include_self_lag = bool(include_self_lag)
        self.self_max_lag = int(self_max_lag)

        self.descendant_adjustment = bool(descendant_adjustment)
        self.anomaly_threshold = float(anomaly_threshold)

        self.normal_window = normal_window
        self.eps = float(eps)

        self.results_: Dict[str, CIRCANodeResult] = {}
        self.raw_scores_: Dict[str, float] = {}
        self.adjusted_scores_: Dict[str, float] = {}
        self.ranking_: List[str] = []


    # ========================================================================
    # Graph utilities
    # ========================================================================

    @staticmethod
    def _directed_edges(graph) -> List[Tuple[str, str]]:
        """
        Return directed graph edges.

        Self-loops are preserved here because they may contain useful
        temporal information. They are removed when constructing the DAG
        used by descendant adjustment.
        """
        return list(graph.get_directed_edges())


    @classmethod
    def _nodes(cls, graph, data: pd.DataFrame) -> List[str]:
        """
        Get graph nodes.

        DataFrame columns are used as the canonical node set because the
        current pyciphod experiments represent each time series by one column.
        """
        nodes = list(data.columns)

        edges = cls._directed_edges(graph)

        graph_nodes = set()
        for source, target in edges:
            graph_nodes.add(source)
            graph_nodes.add(target)

        # Include isolated variables present in data.
        return [
            node for node in nodes
            if node in graph_nodes or node in data.columns
        ]


    @staticmethod
    def _parents(graph, node: str) -> List[str]:
        """
        Return cross-variable parents of node.

        A self-loop Y -> Y in an SCG is not used as a contemporaneous
        predictor. Temporal self-dependence is handled explicitly through
        include_self_lag.
        """
        parents = list(graph.get_parents(node))

        return sorted(
            parent
            for parent in parents
            if parent != node
        )


    @classmethod
    def _children(
        cls,
        graph,
        nodes: Iterable[str],
    ) -> Dict[str, List[str]]:
        """
        Construct cross-variable children.

        Self-loops are removed because CIRCA descendant adjustment operates
        on the cross-variable DAG.
        """
        node_set = set(nodes)

        children = {
            node: []
            for node in node_set
        }

        for source, target in cls._directed_edges(graph):

            if source == target:
                continue

            if source not in node_set or target not in node_set:
                continue

            children[source].append(target)

        for node in children:
            children[node] = sorted(set(children[node]))

        return children


    @classmethod
    def _topological_order(
        cls,
        graph,
        nodes: Sequence[str],
    ) -> List[str]:
        """
        Topological order of the cross-variable graph.

        Self-loops are ignored.

        Raises
        ------
        ValueError
            If the remaining graph contains a directed cycle.
        """
        node_set = set(nodes)

        children = cls._children(
            graph=graph,
            nodes=nodes,
        )

        indegree = {
            node: 0
            for node in node_set
        }

        for source in children:
            for target in children[source]:
                indegree[target] += 1

        queue = sorted(
            node
            for node, degree in indegree.items()
            if degree == 0
        )

        order: List[str] = []

        while queue:

            node = queue.pop(0)

            order.append(node)

            for child in children[node]:

                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)
                    queue.sort()

        if len(order) != len(node_set):
            raise ValueError(
                "CIRCA descendant adjustment requires the cross-variable "
                "graph to be acyclic. The supplied graph contains a "
                "multi-variable directed cycle. You can still run RHT by "
                "setting descendant_adjustment=False."
            )

        return order


    # ========================================================================
    # Regression design
    # ========================================================================

    def _make_design(
        self,
        *,
        data: pd.DataFrame,
        graph,
        target: str,
    ) -> Tuple[pd.DataFrame, Tuple[str, ...], Tuple[str, ...]]:
        """
        Construct the regression DataFrame for

            Y_t = f(Pa(Y_t)) + epsilon_t.

        Returns
        -------
        design:
            DataFrame containing target and predictors.

        parents:
            Cross-variable parents from the supplied graph.

        feature_names:
            Names of all regression predictors.
        """
        if target not in data.columns:
            raise ValueError(
                f"Target {target!r} is not a column of data."
            )

        parents = self._parents(graph, target)

        design = pd.DataFrame(
            index=data.index
        )

        design["__target__"] = data[target]

        feature_names: List[str] = []


        # --------------------------------------------------------------------
        # Graph parents
        # --------------------------------------------------------------------

        for parent in parents:

            if parent not in data.columns:
                raise ValueError(
                    f"Parent {parent!r} of {target!r} is missing "
                    f"from the DataFrame."
                )

            for lag in range(self.parent_max_lag + 1):

                name = f"{parent}[t-{lag}]"

                if lag == 0:
                    design[name] = data[parent]
                else:
                    design[name] = data[parent].shift(lag)

                feature_names.append(name)


        # --------------------------------------------------------------------
        # Autoregressive self-history
        # --------------------------------------------------------------------

        if self.include_self_lag:

            for lag in range(1, self.self_max_lag + 1):

                name = f"{target}[t-{lag}]"

                design[name] = data[target].shift(lag)

                feature_names.append(name)


        return (
            design,
            tuple(parents),
            tuple(feature_names),
        )


    # ========================================================================
    # RHT
    # ========================================================================

    def _score_node(
        self,
        *,
        graph,
        data: pd.DataFrame,
        node: str,
        anomaly_start: int,
        anomaly_size: int,
    ) -> CIRCANodeResult:
        """
        Regression-based hypothesis testing score for one node.
        """

        (
            design,
            parents,
            feature_names,
        ) = self._make_design(
            data=data,
            graph=graph,
            target=node,
        )


        # --------------------------------------------------------------------
        # Define normal and anomalous intervals by POSITION.
        # --------------------------------------------------------------------

        normal_end = anomaly_start

        if self.normal_window is None:
            normal_begin = 0
        else:
            normal_begin = max(
                0,
                anomaly_start - self.normal_window,
            )

        anomaly_end = min(
            len(data),
            anomaly_start + anomaly_size,
        )

        normal = design.iloc[
            normal_begin:normal_end
        ].dropna()

        anomalous = design.iloc[
            anomaly_start:anomaly_end
        ].dropna()


        if normal.empty:
            raise ValueError(
                f"No valid normal observations available for {node!r}."
            )

        if anomalous.empty:
            raise ValueError(
                f"No valid anomalous observations available for {node!r}."
            )


        y_normal = normal["__target__"].to_numpy(
            dtype=float
        )

        y_anomalous = anomalous["__target__"].to_numpy(
            dtype=float
        )


        # --------------------------------------------------------------------
        # Parentless variable:
        #
        # P(Y | Pa(Y)) = P(Y)
        #
        # The intercept-only predictor is simply the normal mean.
        # --------------------------------------------------------------------

        if len(feature_names) == 0:

            prediction_normal = np.full(
                len(y_normal),
                y_normal.mean(),
                dtype=float,
            )

            prediction_anomalous = np.full(
                len(y_anomalous),
                y_normal.mean(),
                dtype=float,
            )


        # --------------------------------------------------------------------
        # Regression on causal parents
        # --------------------------------------------------------------------

        else:

            X_normal = normal[
                list(feature_names)
            ].to_numpy(dtype=float)

            X_anomalous = anomalous[
                list(feature_names)
            ].to_numpy(dtype=float)

            if len(y_normal) <= X_normal.shape[1]:
                raise ValueError(
                    f"Not enough normal observations to fit CIRCA for "
                    f"{node!r}: {len(y_normal)} observations and "
                    f"{X_normal.shape[1]} predictors."
                )

            model = clone(self.regressor)

            model.fit(
                X_normal,
                y_normal,
            )

            prediction_normal = model.predict(
                X_normal
            )

            prediction_anomalous = model.predict(
                X_anomalous
            )


        # --------------------------------------------------------------------
        # Residuals
        # --------------------------------------------------------------------

        residual_normal = (
            y_normal - prediction_normal
        )

        residual_anomalous = (
            y_anomalous - prediction_anomalous
        )


        residual_mean = float(
            np.mean(residual_normal)
        )

        # StandardScaler, used by the reference CIRCA implementation,
        # corresponds to ddof=0.
        residual_std = float(
            np.std(
                residual_normal,
                ddof=0,
            )
        )


        # --------------------------------------------------------------------
        # CIRCA Eq. (3)
        #
        #            residual_t - mu_epsilon
        # a_Y(t) = ---------------------------
        #                 sigma_epsilon
        #
        # --------------------------------------------------------------------

        if not np.isfinite(residual_std):
            raise ValueError(
                f"Non-finite residual standard deviation for {node!r}."
            )

        if residual_std <= self.eps:

            # sklearn's StandardScaler effectively uses scale=1 for a
            # constant training feature. This keeps the computation finite
            # and gives the same practical interpretation here.
            scale = 1.0

        else:
            scale = residual_std

        anomalous_z = (
            residual_anomalous - residual_mean
        ) / scale


        # --------------------------------------------------------------------
        # CIRCA Eq. (4)
        #
        # s_Y = max_t |a_Y(t)|
        #
        # --------------------------------------------------------------------

        raw_score = float(
            np.max(
                np.abs(anomalous_z)
            )
        )


        return CIRCANodeResult(
            node=node,
            raw_score=raw_score,
            adjusted_score=raw_score,
            max_anomalous_z=raw_score,
            parents=parents,
            features=feature_names,
            normal_residual_mean=residual_mean,
            normal_residual_std=residual_std,
        )


    def regression_hypothesis_test(
        self,
        *,
        graph,
        data: pd.DataFrame,
        anomaly_start: int,
        anomaly_size: Optional[int] = None,
    ) -> Dict[str, CIRCANodeResult]:
        """
        Run CIRCA regression-based hypothesis testing for every variable.
        """

        self._validate_input(
            data=data,
            anomaly_start=anomaly_start,
            anomaly_size=anomaly_size,
        )

        if anomaly_size is None:
            anomaly_size = (
                len(data) - anomaly_start
            )

        nodes = self._nodes(
            graph=graph,
            data=data,
        )

        results = {}

        for node in nodes:

            results[node] = self._score_node(
                graph=graph,
                data=data,
                node=node,
                anomaly_start=anomaly_start,
                anomaly_size=anomaly_size,
            )

        return results


    # ========================================================================
    # Descendant adjustment
    # ========================================================================

    def _descendant_adjustment(
        self,
        *,
        graph,
        results: Dict[str, CIRCANodeResult],
    ) -> Dict[str, CIRCANodeResult]:
        """
        CIRCA Algorithm 2: descendant adjustment.

        Let s(Y) denote the raw RHT score.

        For each Y, collect anomaly scores from its children. If a child has
        score < threshold, propagate scores collected from that child's
        descendants as well.

        For Y with s(Y) >= threshold:

            s'(Y) = s(Y) + max(S(Y))

        whenever S(Y) is non-empty.
        """

        nodes = list(results.keys())

        order = self._topological_order(
            graph=graph,
            nodes=nodes,
        )

        children = self._children(
            graph=graph,
            nodes=nodes,
        )

        raw_scores = {
            node: results[node].raw_score
            for node in nodes
        }


        # S[node] in CIRCA Algorithm 2.
        descendant_scores: Dict[
            str,
            List[float]
        ] = {
            node: []
            for node in nodes
        }


        # --------------------------------------------------------------------
        # Algorithm 2 is traversed from sinks toward roots.
        # --------------------------------------------------------------------

        for node in reversed(order):

            collected: List[float] = []

            for child in children[node]:

                child_score = raw_scores[child]

                # Direct child score.
                collected.append(
                    child_score
                )

                # If child itself is below the 3-sigma threshold,
                # propagate candidate effects farther downstream.
                if child_score < self.anomaly_threshold:
                    collected.extend(
                        descendant_scores[child]
                    )

            descendant_scores[node] = collected


        # --------------------------------------------------------------------
        # Adjustment
        # --------------------------------------------------------------------

        for node in nodes:

            raw = raw_scores[node]
            adjusted = raw

            if (
                raw >= self.anomaly_threshold
                and descendant_scores[node]
            ):
                adjusted += max(
                    descendant_scores[node]
                )

            results[node].adjusted_score = float(
                adjusted
            )

        return results


    # ========================================================================
    # Main API
    # ========================================================================

    def fit(
        self,
        *,
        graph,
        data: pd.DataFrame,
        anomaly_start: int,
        anomaly_size: Optional[int] = None,
    ) -> "CIRCA":
        """
        Run CIRCA.

        Parameters
        ----------
        graph:
            Supplied causal graph.

        data:
            pandas DataFrame. Columns correspond to time series.

        anomaly_start:
            POSITION of the first anomalous observation.

        anomaly_size:
            Number of anomalous observations to use. If None, use all
            observations from anomaly_start onward.
        """

        results = self.regression_hypothesis_test(
            graph=graph,
            data=data,
            anomaly_start=anomaly_start,
            anomaly_size=anomaly_size,
        )


        if self.descendant_adjustment:

            results = self._descendant_adjustment(
                graph=graph,
                results=results,
            )


        self.results_ = results

        self.raw_scores_ = {
            node: result.raw_score
            for node, result in results.items()
        }

        self.adjusted_scores_ = {
            node: result.adjusted_score
            for node, result in results.items()
        }


        # --------------------------------------------------------------------
        # CIRCA is fundamentally a ranking method.
        # --------------------------------------------------------------------

        self.ranking_ = sorted(
            results,
            key=lambda node: (
                results[node].adjusted_score,
                results[node].raw_score,
            ),
            reverse=True,
        )

        return self


    def rank(self) -> List[str]:
        """
        Return variables ranked from most to least suspicious.
        """
        self._check_fitted()

        return list(self.ranking_)


    def get_scores(
        self,
        *,
        adjusted: bool = True,
    ) -> Dict[str, float]:
        """
        Return node -> CIRCA score.
        """
        self._check_fitted()

        scores = (
            self.adjusted_scores_
            if adjusted
            else self.raw_scores_
        )

        return dict(scores)


    def get_results(self) -> pd.DataFrame:
        """
        Detailed ranked results as a DataFrame.
        """
        self._check_fitted()

        rows = []

        for rank, node in enumerate(
            self.ranking_,
            start=1,
        ):

            result = self.results_[node]

            rows.append(
                {
                    "rank": rank,
                    "node": node,
                    "score": result.adjusted_score,
                    "raw_score": result.raw_score,
                    "adjusted_score": result.adjusted_score,
                    "parents": result.parents,
                    "features": result.features,
                    "normal_residual_mean":
                        result.normal_residual_mean,
                    "normal_residual_std":
                        result.normal_residual_std,
                }
            )

        return pd.DataFrame(rows)


    def predict_top_k(
        self,
        k: int,
    ) -> List[str]:
        """
        Return the k highest-ranked variables.
        """
        self._check_fitted()

        if k <= 0:
            return []

        return self.ranking_[:k]


    def predict_threshold(
        self,
        threshold: Optional[float] = None,
        *,
        use_raw_score: bool = True,
    ) -> List[str]:
        """
        Return variables whose CIRCA score is at least threshold.

        By default, the raw RHT score is used because CIRCA's three-sigma
        criterion is defined before descendant adjustment.
        """
        self._check_fitted()

        if threshold is None:
            threshold = self.anomaly_threshold

        if use_raw_score:

            return [
                node
                for node in self.ranking_
                if self.results_[node].raw_score >= threshold
            ]

        return [
            node
            for node in self.ranking_
            if self.results_[node].adjusted_score >= threshold
        ]


    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_input(
        *,
        data: pd.DataFrame,
        anomaly_start: int,
        anomaly_size: Optional[int],
    ) -> None:

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "data must not be empty."
            )

        if not isinstance(anomaly_start, (int, np.integer)):
            raise TypeError(
                "anomaly_start must be an integer row position."
            )

        if anomaly_start <= 0:
            raise ValueError(
                "anomaly_start must be strictly positive because "
                "CIRCA needs normal observations before the anomaly."
            )

        if anomaly_start >= len(data):
            raise ValueError(
                "anomaly_start must be smaller than len(data)."
            )

        if anomaly_size is not None:

            if anomaly_size <= 0:
                raise ValueError(
                    "anomaly_size must be positive."
                )


    def _check_fitted(self) -> None:

        if not self.results_:
            raise RuntimeError(
                "CIRCA has not been fitted yet. Call fit() first."
            )