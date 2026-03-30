import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional, Any, Tuple, Union

from pyciphod.utils.graphs.graphs import DirectedAcyclicGraph, AcyclicDirectedMixedGraph
from pyciphod.utils.graphs.graphs import create_random_dag, create_random_admg


class SCM:
    """
    Structural Causal Model.

    Parameters
    ----------
    v : iterable of str
        Endogenous (observed) variable names.
    u : iterable of str
        Exogenous (unobserved - latent) variable names.
    f : dict
        Mapping variable -> {'parents': list_of_parent_names, 'func': callable}.
        Each callable must have signature func(parents_dict, noise) and return a scalar.
        If a variable is missing from `f`, it will be treated as exogenous noise (identity of noise).
    u_dist : None, callable or dict
        If None, Gaussian N(0,1) is used for all noise. If a callable, it is used for all variables
        and must accept either no arguments (return scalar) or a keyword `size` to produce an array.
        If a dict, keys are variable names and values are callables producing noise for that variable.
    """

    def __init__(self, v: Iterable[str], u: Iterable[str], f: Dict[str, Dict[str, Any]], u_dist: Optional[Any] = None):
        self.v: List[str] = list(v)
        self.u: List[str] = list(u)
        self._all_vars: List[str] = self.u + self.v
        self.f = f or {}

        # normalize u_dist into a mapping var -> callable
        if u_dist is None:
            # Use None as sentinel to indicate "use internal rng" in generate_data
            self._u_dist_map = {name: None for name in self._all_vars}
        elif callable(u_dist):
            self._u_dist_map = {name: u_dist for name in self._all_vars}
        elif isinstance(u_dist, dict):
            # fallback: missing entries -> std normal
            # missing entries -> use internal rng (None sentinel)
            self._u_dist_map = {name: (u_dist.get(name, None)) for name in self._all_vars}
        else:
            raise ValueError("u_dist must be None, callable or dict mapping variable->callable")

        # build DAG using DirectedAcyclicGraph from graphs.py
        self._dag = DirectedAcyclicGraph()
        for name in self._all_vars:
            self._dag.add_vertex(name)

        # Add directed edges according to f's parents specification
        for var, spec in (self.f.items() if isinstance(self.f, dict) else []):
            parents = spec.get('parents', []) if isinstance(spec, dict) else []
            for p in parents:
                # Do not allow observed -> unobserved (endogenous causing exogenous)
                if p in self.v and var in self.u:
                    raise ValueError(f"Invalid structural equation: observed variable '{p}' cannot be a parent of latent variable '{var}'")
                # DirectedAcyclicGraph.add_directed_edge will raise ValueError if cycle would be created
                self._dag.add_directed_edge(p, var)

    def induced_dag(self) -> DirectedAcyclicGraph:
        """
        Return the DirectedAcyclicGraph induced by the SCM (observed + latent variables).

        If there is unobserved confounding (a latent that is an ancestor of two or more observed
        variables), raise ValueError and suggest using `induced_admg()` which can represent
        such confounding via bidirected edges.
        """
        # Check for latent-induced confounding: any latent with >=2 observed descendants
        for latent in self.u:
            obs_children = set(self._dag.get_children(latent)).intersection(self.v)
            if len(obs_children) >= 2:
                raise ValueError(
                    f"Unobserved confounding detected: latent '{latent}' is an ancestor of observed variables {sorted(obs_children)}. "
                    "Use `induced_admg()` to obtain an ADMG that represents such confounding."
                )

        # Build a DAG containing only the observed variables and the directed edges among them
        new_dag = DirectedAcyclicGraph()
        for ov in self.v:
            new_dag.add_vertex(ov)

        for (src, tgt) in self._dag.get_directed_edges():
            if src in self.v and tgt in self.v:
                new_dag.add_directed_edge(src, tgt)

        return new_dag

    def induced_admg(self) -> AcyclicDirectedMixedGraph:
        """
        Return an AcyclicDirectedMixedGraph induced by the SCM over the observed variables.

        Directed edges among observed variables are preserved. Latent variables that are
        ancestors of two or more observed variables are converted into bidirected (confounded)
        edges between every pair of their observed descendants. If multiple latents induce the
        same confounding, the confounded edge is added once.
        """
        admg = AcyclicDirectedMixedGraph()
        # add observed vertices
        for ov in self.v:
            admg.add_vertex(ov)

        # preserve directed edges among observed variables
        for (src, tgt) in self._dag.get_directed_edges():
            if src in self.v and tgt in self.v:
                admg.add_directed_edge(src, tgt)

        # convert latent ancestors into bidirected confounding among their observed descendants
        for latent in self.u:
            obs_children = set(self._dag.get_children(latent)).intersection(self.v)
            obs_list = sorted(obs_children)
            for i in range(len(obs_list)):
                for j in range(i + 1, len(obs_list)):
                    a, b = obs_list[i], obs_list[j]
                    # add a bidirected/confounded edge between a and b
                    admg.add_confounded_edge(a, b)

        return admg

    def generate_data(self, n_samples: int = 1000, include_latent: bool = False, seed: Optional[int] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Simulate data from the SCM.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        include_latent : bool
            If True, include latent variables `u` in the returned data (as a separate DataFrame tuple).
            If False (default), only observed variables `v` are returned.
        seed : Optional[int]
            Seed for the random number generator (numpy).

        Returns
        -------
        pd.DataFrame or tuple(pd.DataFrame, pd.DataFrame)
            If include_latent=False: returns a DataFrame with observed variables `v`.
            If include_latent=True: returns a tuple (df_observed, df_latent).
        """
        rng = np.random.default_rng(seed)

        # Topological order using the internal networkx DiGraph
        topo_order = list(nx.topological_sort(self._dag._directed_g))

        # storage
        data = {var: np.empty(n_samples, dtype=float) for var in self._all_vars}

        # Evaluate each sample following topological order
        for i in range(n_samples):
            for var in topo_order:
                # determine parent values for this sample
                parents = []
                if var in self.f and isinstance(self.f[var], dict):
                    parents = self.f[var].get('parents', [])
                parents_vals = {p: data[p][i] for p in parents}

                # sample noise for this variable using the mapped sampler
                sampler = self._u_dist_map.get(var)
                if sampler is None:
                    noise = float(rng.normal())
                else:
                    # try to call sampler with a size argument, then fallback to no-arg call
                    try:
                        noise_arr = sampler(size=1)
                        noise = float(np.asarray(noise_arr).reshape(-1)[0])
                    except TypeError:
                        try:
                            noise = float(sampler())
                        except Exception:
                            noise = float(rng.normal())
                    except Exception:
                        noise = float(rng.normal())

                # compute the structural function or treat as exogenous noise
                if var in self.f and isinstance(self.f[var], dict) and callable(self.f[var].get('func')):
                    val = self.f[var]['func'](parents_vals, noise)
                else:
                    val = noise

                data[var][i] = val

        # Build DataFrame for observed variables
        df_observed = pd.DataFrame({col: data[col] for col in self.v})

        if include_latent:
            # build separate DataFrame for latent variables and return both
            df_latent = pd.DataFrame({col: data[col] for col in self.u})
            return df_observed, df_latent

        return df_observed


# Convenience small helpers
class LinearSCM(SCM):
    """A simple Linear SCM helper that accepts a coefficient-based specification."""

    def __init__(self, v: Iterable[str], u: Iterable[str], coefficients: Dict[tuple, float], intercepts: Optional[Dict[str, float]] = None, u_dist: Optional[Any] = None):
        # build f from coefficients: for each target, collect incoming parents
        coeffs = coefficients or {}
        intercepts = intercepts or {}
        all_vars = list(u) + list(v)

        # Build parent lists and linear functions
        f = {}
        parents_map: Dict[str, List[str]] = {var: [] for var in all_vars}
        for (src, tgt), w in coeffs.items():
            # Validate coefficients do not create observed -> latent edges
            if src in v and tgt in u:
                raise ValueError(f"Invalid coefficient specification: observed variable '{src}' cannot be a parent of latent variable '{tgt}'")
            parents_map[tgt].append(src)

        for var in all_vars:
            parents = parents_map.get(var, [])
            b = float(intercepts.get(var, 0.0))

            def make_linear_func(var, parents, b):
                def func(par_vals, noise):
                    s = b
                    for p in parents:
                        s += par_vals.get(p, 0.0) * float(coeffs.get((p, var), 0.0))
                    s += noise
                    return s
                return func

            f[var] = {'parents': parents, 'func': make_linear_func(var, parents, b)}

        super().__init__(v=v, u=u, f=f, u_dist=u_dist)


def create_random_linear_scm_from_admg(admg: AcyclicDirectedMixedGraph,
                                      weight_low: float = -1.0,
                                      weight_high: float = 1.0,
                                      intercept_low: float = 0.0,
                                      intercept_high: float = 0.0,
                                      u_dist: Optional[Any] = None,
                                      seed: Optional[int] = None):
    """
    Build a LinearSCM from an existing DirectedAcyclicGraph by sampling random edge weights
    and intercepts.

    Returns (scm, coefficients, intercepts)
    - scm : LinearSCM instance
    - coefficients : dict mapping (src, tgt) -> weight
    - intercepts : dict mapping var -> intercept
    """
    rng = np.random.default_rng(seed)


    coefficients: Dict[tuple, float] = {}
    # Generate unobserved variables and generate random coefficients between unobserved and observed and store in a dict
    u = []
    v_in_conf = []
    for (tgt_1, tgt_2) in sorted(admg.get_confounded_edges()):
        src = f"U_{tgt_1}_{tgt_2}"
        coefficients[(src, tgt_1)] = 1
        coefficients[(src, tgt_2)] = 1
        u = u + [src]
        v_in_conf += [tgt_1, tgt_2]

    v= sorted(admg.get_vertices())
    for tgt in v:
        if tgt not in v_in_conf:
            src = f"U_{tgt}"
            coefficients[(src, tgt)] = 1
            u = u + [src]

    # collect directed edges from the ADMG to assign random weights
    directed_edges = sorted(admg.get_directed_edges())

    # Generate random coefficients for all edges between observed and store in a dict
    for (src, tgt) in directed_edges:
        w = float(rng.uniform(weight_low, weight_high))
        coefficients[(src, tgt)] = w

    # generate intercepts for all variables (latent + observed)
    all_vars = list(u) + list(v)
    intercepts: Dict[str, float] = {var: float(rng.uniform(intercept_low, intercept_high)) for var in all_vars}

    scm = LinearSCM(v=list(v), u=list(u), coefficients=coefficients, intercepts=intercepts, u_dist=u_dist)
    return scm, coefficients, intercepts


def create_random_linear_scm_from_dag(dag: DirectedAcyclicGraph,
                                      weight_low: float = -1.0,
                                      weight_high: float = 1.0,
                                      intercept_low: float = 0.0,
                                      intercept_high: float = 0.0,
                                      u_dist: Optional[Any] = None,
                                      seed: Optional[int] = None):
    admg = dag
    return create_random_linear_scm_from_admg(admg, weight_low,
                                             weight_high, intercept_low, intercept_high,
                                             u_dist, seed)


def create_random_linear_scm(num_v: int,
                             p_edge: float = 0.2,
                             weight_low: float = -1.0,
                             weight_high: float = 1.0,
                             intercept_low: float = 0.0,
                             intercept_high: float = 0.0,
                             unmeasured_confounding: bool = False,
                             u_dist: Optional[Any] = None,
                             seed: Optional[int] = None):
    """
    Generate a random acyclic DAG and build a LinearSCM on top of it.

    Parameters
    ----------
    num_v : int
        Number of observed variables (v). Latent variables are created internally.
    p_edge : float
        Probability to add an edge between two nodes (from earlier to later in a random order).
    weight_low, weight_high : float
        Interval to sample edge weights.
    intercept_low, intercept_high : float
        Interval to sample variable intercepts.
    u_dist : None|callable|dict
        Noise distribution(s) passed to LinearSCM.
    seed : int | None
        Seed for reproducibility.

    Returns (scm, dag, coefficients, intercepts)
    """
    rng = np.random.default_rng(seed)

    # name variables: U0..U{num_u-1}, X0..X{num_v-1}
    # u = [f"U{i}" for i in range(num_v)]
    if unmeasured_confounding:
        admg = create_random_admg(num_v=num_v, p_edge=p_edge, seed=seed)
    else:
        admg = create_random_dag(num_v=num_v, p_edge=p_edge, seed=seed)

    scm, coefficients, intercepts = create_random_linear_scm_from_admg(admg, weight_low, weight_high, intercept_low, intercept_high, u_dist, seed)
    return scm,  coefficients, intercepts


if __name__ == '__main__':
    # def univ():
    #     return np.random.uniform(0.51, 1)
    scm, coefficients, intercepts = create_random_linear_scm(num_v=3, p_edge=0.6, seed=0, unmeasured_confounding=True)
    df_obs, df_lat = scm.generate_data(n_samples=5, include_latent=True, seed=0)
    dag = scm.induced_admg()
    print(dag.get_confounded_edges())
    print('DAG directed edges:', dag.get_directed_edges())
    print('Coefficients:', coefficients)
    print('\nObserved data:\n', df_obs)
    print('\nLatent data:\n', df_lat)
    print('\nExample function for X0:', scm.f.get('X0', {}).get('func'))
    print(scm.f)
    dag.draw_graph()
