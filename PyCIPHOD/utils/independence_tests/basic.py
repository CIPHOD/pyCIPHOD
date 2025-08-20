import numpy as np
from scipy.stats import norm

from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression as fr


from causallearn.utils.cit import CIT

def fisherz_CI_test(data, X, Y, S, alpha = 0.05) -> bool:
    """
    Test conditional independence with Fisher Z test using CIT.

    Args:
        data: pandas DataFrame with the variables.
        X (str): name of first variable.
        Y (str): name of second variable.
        S (list[str] or None): conditioning set variable names.
        alpha (float): significance threshold.

    Returns:
        bool: True if X _independent_ of Y given S (p > alpha), else False.
    """
    # Convert DataFrame to numpy array
    data_np = data.to_numpy()

    # Map variable names to indices in the numpy array
    var_idx = {var: idx for idx, var in enumerate(data.columns)}

    fisherz_obj = CIT(data_np, "fisherz")

    # Convert variable names X, Y, S to indices
    x_idx = var_idx[X]
    y_idx = var_idx[Y]
    s_idx = [var_idx[s] for s in S] if S else []

    pValue = fisherz_obj(x_idx, y_idx, s_idx)

    return pValue 

def gsq_CI_test(data, X, Y, S, alpha=0.05) -> bool:
    """
    Test conditional independence with G² (likelihood ratio chi-squared) test using CIT.

    Args:
        data: pandas DataFrame with the variables (assumed categorical, ideally binary).
        X (str): name of first variable.
        Y (str): name of second variable.
        S (list[str] or None): conditioning set variable names.
        alpha (float): significance threshold.

    Returns:
        bool: True if X _independent_ of Y given S (p > alpha), else False.
    """
    # Convert DataFrame to numpy array
    data_np = data.to_numpy()

    # Map variable names to indices in the numpy array
    var_idx = {var: idx for idx, var in enumerate(data.columns)}

    # Create G² CIT instance
    gsq_obj = CIT(data_np, "gsq")

    # Convert variable names to indices
    x_idx = var_idx[X]
    y_idx = var_idx[Y]
    s_idx = [var_idx[s] for s in S] if S else []

    # Run the G² test
    pValue = gsq_obj(x_idx, y_idx, s_idx)

    return pValue 

def chi2_CI_test(data, X, Y, S, alpha=0.05) -> bool:
    """
    Test conditional independence with Pearson Chi-squared test using CIT.

    Args:
        data: pandas DataFrame with the variables (assumed categorical, ideally binary).
        X (str): name of first variable.
        Y (str): name of second variable.
        S (list[str] or None): conditioning set variable names.
        alpha (float): significance threshold.

    Returns:
        bool: True if X _independent_ of Y given S (p > alpha), else False.
    """
    # Convert DataFrame to numpy array
    data_np = data.to_numpy()

    # Map variable names to indices in the numpy array
    var_idx = {var: idx for idx, var in enumerate(data.columns)}

    # Create Chi-squared CIT instance
    chi2_obj = CIT(data_np, "chisq")

    # Convert variable names to indices
    x_idx = var_idx[X]
    y_idx = var_idx[Y]
    s_idx = [var_idx[s] for s in S] if S else []

    # Run the chi-squared test
    pValue = chi2_obj(x_idx, y_idx, s_idx)

    return pValue 

def kci_CI_test(data, X, Y, S, alpha=0.05) -> bool:
    """
    Test conditional independence using the Kernel-based Conditional Independence (KCI) test.

    Args:
        data: pandas DataFrame with the variables (continuous or mixed-type).
        X (str): name of first variable.
        Y (str): name of second variable.
        S (list[str] or None): conditioning set variable names.
        alpha (float): significance threshold.

    Returns:
        bool: True if X _independent_ of Y given S (p > alpha), else False.
    """
    # Convert DataFrame to numpy array
    data_np = data.to_numpy()

    # Map variable names to indices in the numpy array
    var_idx = {var: idx for idx, var in enumerate(data.columns)}

    # Create KCI CIT instance
    kci_obj = CIT(data_np, "kci")

    # Convert variable names to indices
    x_idx = var_idx[X]
    y_idx = var_idx[Y]
    s_idx = [var_idx[s] for s in S] if S else []

    # Run the KCI test
    pValue = kci_obj(x_idx, y_idx, s_idx)

    return pValue 





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
    def __init__(self, x, y, cond_list=None):
        CiTests.__init__(self, x, y, cond_list)

    def get_dependence(self, df):
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
        r = self.get_dependence(df)
        if r == 1:
            r = r - 0.0000000001
        z = 0.5 * np.log((1 + r) / (1 - r))
        pval = np.sqrt(df.shape[0] - len(self.cond_list) - 3) * abs(z)
        pval = 2 * (1 - norm.cdf(abs(pval)))

        return pval, r


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
