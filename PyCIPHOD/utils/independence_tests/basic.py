import pandas as pd
import numpy as np
from scipy.stats import norm, chi2

from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression as fr

class CiTests:
    def __init__(self, x, y, cond_list=None):
        super(CiTests, self).__init__()
        self.x = x
        self.y = y
        if cond_list is None:
            self.cond_list = []
        else:
            self.cond_list = cond_list

    def get_dependence(self, df):
        print("To be implemented")

    def get_pvalue(self, df):
        print("To be implemented")


class FisherZ(CiTests):
    """
    Continuous conditional independence test using the Fisher Z-transform.
    Suitable for Gaussian data. Computes partial correlations and corresponding p-values.
    Reference: Spirtes et al., "Causation, Prediction, and Search".
    """

    def __init__(self, x, y, cond_list=None):
        """
        Initialize the test with variables X, Y and optional conditioning set Z.
        :param x: str, name of first variable
        :param y: str, name of second variable
        :param cond_list: list of str, conditioning variables
        """
        super().__init__(x, y, cond_list)

    def get_dependence(self, df):
        """
        Compute the partial correlation between X and Y given Z.
        Returns a correlation coefficient r in [-1,1].
        """
        list_nodes = [self.x, self.y] + self.cond_list
        df = df[list_nodes]
        a = df.values.T

        if len(self.cond_list) > 0:
            cond_list_int = [i + 2 for i in range(len(self.cond_list))]
        else:
            cond_list_int = []

        correlation_matrix = np.corrcoef(a)
        var = list((0, 1) + tuple(cond_list_int))
        sub_corr_matrix = correlation_matrix[np.ix_(var, var)]

        if np.linalg.det(sub_corr_matrix) == 0:
            r = 1
        else:
            inv = np.linalg.inv(sub_corr_matrix)
            r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

        return r

    def get_pvalue(self, df):
        """
        Compute the p-value for the partial correlation.
        Uses the Fisher Z-transform:
            z = 0.5 * ln((1+r)/(1-r))
            test statistic = sqrt(n - |Z| - 3) * |z|
        Returns a two-sided p-value.
        """
        r = self.get_dependence(df)
        # Avoid numerical issues for r=1
        if r == 1:
            r -= 1e-10
        z = 0.5 * np.log((1 + r) / (1 - r))
        test_stat = np.sqrt(df.shape[0] - len(self.cond_list) - 3) * abs(z)
        pval = 2 * (1 - norm.cdf(abs(test_stat)))
        return pval


class Gsq(CiTests):
    """
    Discrete conditional independence test using the G² likelihood-ratio test.
    Reference: Spirtes et al., "Causation, Prediction, and Search".
    """

    def __init__(self, x, y, cond_list=None):
        super().__init__(x, y, cond_list)

    def _contingency_table(self, df):
        """
        Compute the joint frequency table for X, Y given Z.
        Returns a dictionary mapping conditioning states to 2D contingency arrays.
        """
        if not self.cond_list:
            # No conditioning: simple 2D table
            table = pd.crosstab(df[self.x], df[self.y])
            return {(): table.values}
        else:
            # Group by conditioning variables
            grouped = df.groupby(self.cond_list)
            tables = {}
            for cond_values, sub_df in grouped:
                table = pd.crosstab(sub_df[self.x], sub_df[self.y])
                tables[cond_values] = table.values
            return tables

    def get_dependence(self, df):
        """
        Returns the G² statistic.
        """
        tables = self._contingency_table(df)
        g2_stat = 0
        for table in tables.values():
            if table.size == 0:
                continue
            n = np.sum(table)
            row_sums = np.sum(table, axis=1)
            col_sums = np.sum(table, axis=0)
            expected = np.outer(row_sums, col_sums) / n
            # Only consider nonzero expected counts
            mask = expected > 0
            g2_stat += 2 * np.sum(table[mask] * np.log(table[mask] / expected[mask]))
        return g2_stat

    def get_pvalue(self, df):
        """
        Returns the p-value of the G² test.
        Degrees of freedom = (|X|-1)*(|Y|-1)*prod(|Z_i|) for each conditioning state.
        """
        tables = self._contingency_table(df)
        g2_stat = self.get_dependence(df)
        # Compute df
        df_total = 0
        for table in tables.values():
            r, c = table.shape
            df_total += (r - 1) * (c - 1)
        pval = 1 - chi2.cdf(g2_stat, df_total)
        return pval





class LinearRegression:
    def __init__(self, x, y, cond_list=[]):
        self.x = x
        self.y = y
        self.list_nodes = [x] + cond_list

    def get_coeff(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        reg = lr().fit(X_data, Y_data)

        return reg.coef_[0]

    def test_zeo_coef(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        pval = fr(X_data, Y_data)[1][0]
        return pval
