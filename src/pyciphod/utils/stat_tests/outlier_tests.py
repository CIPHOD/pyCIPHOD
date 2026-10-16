import numpy as np
from scipy.stats import norm

import scipy as scp


# def grubb_test(quatification_list, sig_threshold=0.05):
#     n = len(quatification_list)
#
#     stat_value = max(abs(np.array(quatification_list) - np.mean(quatification_list)))/ np.std(quatification_list)
#
#     t_dist = scp.stats.t.ppf(1 - sig_threshold / (2 * n), n - 2)
#     numerator = (n - 1) * np.sqrt(np.square(t_dist))
#     denominator = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_dist))
#     critical_value = numerator / denominator
#
#     quatification_list_sorted_idx = sorted(range(len(quatification_list)), key=quatification_list.__getitem__)
#     quatification_list_sorted = sorted(quatification_list)
#     # locate potential outlier
#     l1 = quatification_list_sorted[1] - quatification_list_sorted[0]
#     l2 = quatification_list_sorted[-1] - quatification_list_sorted[-2]
#
#     anomaly_position = None
#     if stat_value > critical_value:
#         if l1 > l2:
#             anomaly_position = quatification_list_sorted_idx[0]
#         else:
#             anomaly_position = quatification_list_sorted_idx[-1]
#
#     return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value}


def grubb_test(quantification_list, sig_threshold=0.05):
    values = np.asarray(quantification_list, dtype=float)
    n = len(values)

    if n < 3:
        raise ValueError("Grubbs' test requires at least 3 observations.")

    std = np.std(values, ddof=1)

    if np.isclose(std, 0):
        return {"anomaly_position": None, "stat_value": 0.0, "critical_value": np.inf,}

    stat_value = np.max(np.abs(values - np.mean(values)))/ std

    t_value = scp.stats.t.ppf(1 - sig_threshold / (2 * n), n - 2,)

    critical_value = (n - 1)/ np.sqrt(n) * np.sqrt(t_value**2 / (n - 2 + t_value**2))

    min_idx = np.argmin(values)
    max_idx = np.argmax(values)

    lower_gap = np.partition(values, 1)[1] - values[min_idx]

    upper_gap = values[max_idx] - np.partition(values, n - 2)[n - 2]

    anomaly_position = None

    if stat_value > critical_value:
        if lower_gap > upper_gap:
            anomaly_position = int(min_idx)
        else:
            anomaly_position = int(max_idx)

    return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value,}