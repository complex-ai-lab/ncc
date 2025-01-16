from typing import Callable, Dict, List
import numpy as np
import properscoring as ps
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

EPS = 1e-6


def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def norm_rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    try:
        scale = MinMaxScaler()
        targets = scale.fit_transform(targets[:,None])
        predictions = scale.transform(predictions[:,None])
    finally:
       return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):
    """
    Mean Absolute Percentage Error
    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model
    Returns:
        float: MAPE
    """
    targets[targets==0] = np.nan
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100


# target ground truth
# mean -
def crps(mean, std, targets):
    """
    Quantile-based CRPS
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: CRPS
    """
    std_clip = np.clip(std, EPS, None)
    ans1 = ps.crps_gaussian(targets, mean, std_clip)
    ans2 = np.abs(targets - mean)
    ans = (~np.isclose(std, 0, atol=1e-6)) * ans1 + (np.isclose(std, 0, atol=1e-6)) * ans2
    return ans.mean()


# -1
def crps_samples(samples, targets):
    """
    Quantile-based CRPS
    Args:
        samples (np.ndarray): Samples of the distribution (N, samples)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: CRPS
    """
    return ps.crps_ensemble(targets, samples).mean()


def log_score(mean, std, targets, window=0.1):
    """
    Log Score
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: Log Score
    """

    # rescale, changed code
    std = np.clip(std, EPS, None)
    scale = MinMaxScaler()
    targets = scale.fit_transform(targets)
    mean = scale.transform(mean)
    std = scale.scale_ * std

    t1 = norm.cdf(targets - window / 2.0, mean, std)
    t2 = norm.cdf(targets + window / 2.0, mean, std)
    a = np.log(np.clip(t2 - t1, EPS, 1.0)).mean()
    return np.clip(a, -10, 10)


# put in slack
def interval_score(mean, std, targets, window=1.0):
    """
    Interval Score
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: Interval Score
    """

    # rescale, changed code
    std = np.clip(std, EPS, None)
    scale = MinMaxScaler()
    targets = scale.fit_transform(targets)
    mean = scale.transform(mean)
    std = scale.scale_ * std

    rd_val = np.round(targets, decimals=1)
    low_val = np.clip(rd_val - window / 2, a_min=0.0, a_max=None)
    high_val = np.clip(rd_val + window / 2, a_min=None, a_max=13)
    t1 = norm.cdf(low_val, loc=mean, scale=std)
    t2 = norm.cdf(high_val, loc=mean, scale=std)
    return np.log(np.clip(t2 - t1, a_min=EPS, a_max=1.0)).mean()


def conf_interval(mean, var, conf):
    """
    Confintance Interval for given confidence level
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        conf (float): Confidence level
    Returns:
        tuple: (low, high) interval
    """
    var = np.clip(var, EPS, None)
    out_prob = 1.0 - conf
    high = norm.ppf(1.0 - (out_prob / 2), loc=mean, scale=var**0.5)
    low = norm.ppf((1.0 - conf) / 2, loc=mean, scale=var**0.5)
    return low, high


def pres_recall(mean, var, target, conf):
    """
    Fraction of GT points within the confidence interval
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        conf (float): Confidence level
    Returns:
        np.ndarray: Fraction of GT points within the confidence interval
    """
    low, high = conf_interval(mean, var, conf)
    truth = ((target > low) & (target < high)).astype("float32")
    return truth.mean(-1)


# Plot
def get_pr(pred, var, target, color="blue", label="FluFNP"):
    """
    Plot confidence and return Confidence score and AUC
    Args:
        pred (np.ndarray): Predictions of the model (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        color (str): Color of the line
        label (str): Label of the model
    Returns:
        tuple: (Confidence score, AUC, fraction values)
    """ 
    pred_, var_, target_ = pred.squeeze(), var.squeeze(), target.squeeze()
    x = np.arange(0.05, 1.0, 0.01)
    y = np.array([pres_recall(pred_, var_, target_, c) for c in x])
    #     plt.plot(list(x) + [1.0], list(y) + [1.0], label=label, color=color)
    conf_score = np.abs(y - x).sum() * 0.01
    auc = y.sum() * 0.01
    return auc, conf_score, list(y) + [1.0]


def dist_from_quantiles(quantiles: Dict[float, float]) -> Callable[[float], float]:
    """
    Returns an approximation cdf for a given quantile
    Args:
        quantiles (Dict[float, float]): Quantiles and corresponding values
    Returns:
        Callable: Approximation cdf
    """

    def cdf(x: float) -> float:
        for q, v in quantiles.items():
            if x < v:
                return q
        return 1.0

    return cdf


def crps_integrated(quantiles: List[Dict[float, float]], target: np.ndarray) -> float:
    """
    Returns CRPS for a given quantile
    Args:
        quantiles (Dict[float, float]): Quantiles and corresponding values
        target (float): Target of the model
    Returns:
        float: CRPS
    """
    cdfs = [dist_from_quantiles(q) for q in quantiles]
    # TODO: This is veeery slow. Can we do better?
    return np.array(
        [
            ps.crps_quadrature([t], cdf, tol=1e-3)
            for cdf, t in zip(cdfs, target.flatten())
        ]
    ).mean()


# ############## #
# New Metrics    #
# ############## #

def calculate_stds_helper(y_pred, lower):
    val_prob = {}
    alphas = sorted(list(lower.keys()))
    for i in range(len(alphas)):
        if i == len(alphas) - 1:
            alpha1 = alphas[i]
            itv1 = y_pred - lower[alpha1]
            val_prob[y_pred] = 1 - alpha1
        else:
            alpha1 = alphas[i]
            alpha2 = alphas[i + 1]
            itv1 = y_pred - lower[alpha1]
            itv2 = y_pred - lower[alpha2]
            val_prob[y_pred + 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
            val_prob[y_pred - 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
    variance = 0
    total_prob = 0
    for val, prob in val_prob.items():
        total_prob += prob
        variance += (val - y_pred) ** 2 * prob
    return variance ** 0.5


def calculate_stds(y_preds, lowers):
    stds = []
    # take one week from data
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        current_std = calculate_stds_helper(y_pred, lower)
        stds.append(current_std)
    return stds


def norm_crps(y_preds, y_trues, lowers):
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    stds = calculate_stds(y_preds, lowers)
    crps_val = crps(y_preds, stds, y_trues)
    return crps_val


def interval_score(y_true, upper, lower, alpha):
    return (upper - lower) + 2 / alpha * (lower - y_true) * (y_true < lower) + 2 / alpha * (y_true - upper) * (upper < y_true)


def weighted_IS_helper(y_pred, y_true, lower):
    # alphas = sorted(list(lower.keys()), reverse=True)[:-1]
    alphas = sorted(list(lower.keys()), reverse=True)
    K = len(alphas)
    sum_term = 0
    for i in range(len(alphas)):
        alpha = alphas[i]
        upper = y_pred - lower[alpha] + y_pred
        sum_term += alphas[i] / 2 * interval_score(y_true, upper, lower[alpha], alpha)
    return 1 / (K + 1/2) * (alphas[0] / 2 * abs(y_true - y_pred) + sum_term)


def weighted_IS(y_preds, y_trues, lowers):
    weighted_IS_vals = []
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        y_true = y_trues[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        weighted_IS_vals.append(weighted_IS_helper(y_pred, y_true, lower))
    return weighted_IS_vals


def monotonicity(y_preds, lowers):
    """
    Measure monotinicity: for a list x of length l. counter=0. For i from 0 to l, if x[i] > any other element behind, counter++. counter/l
    """
    num_weeks = len(y_preds)
    alphas = list(lowers.keys())
    mono_metrics = []
    for i in range(num_weeks):
        y_pred = y_preds[i]
        q_hats = []
        counter = 0
        for alpha in alphas:
            q_hats.append(y_pred - lowers[alpha][i])
        # validate
        # prev_qhat = q_hats[0]
        # for qhat in q_hats:
        #     assert qhat >= 0
        #     assert qhat <= prev_qhat
        #     prev_qhat = qhat
        for k, q_hat in enumerate(q_hats):
            greater_than_right = True
            for j_ in range(len(q_hats)-k-1):
                j = j_ + k + 1
                if q_hat < q_hats[j]:
                    greater_than_right = False
                    break
            if greater_than_right:
                counter += 1
        mono_metrics.append(counter / len(alphas))
    # for item in mono_metrics:
    #     assert item == 1
    return np.mean(mono_metrics)


def validity_score(y_preds, lowers):
    """
    Measure the percentage of valid distributions: for each time step, if its prediction intervals are monotonic w.r.t the confidence levels, then this is a valid distribution.
    """
    num_weeks = len(y_preds)
    alphas = list(lowers.keys())
    num_valid_points = 0
    for i in range(num_weeks):
        y_pred = y_preds[i]
        q_hats = []
        for alpha in alphas:
            q_hats.append(y_pred - lowers[alpha][i])
        valid_point = 1
        for k in range(len(q_hats)-1):
            if q_hats[k] < q_hats[k+1]:
                valid_point = 0
                break
        num_valid_points = num_valid_points + valid_point
    return num_valid_points / num_weeks


def interval_length(y_preds, lowers, target_error_rate=0.1, only_valid=False, y_trues=None, uppers=None):
    target_lower = lowers[target_error_rate]
    half_PI_lengths = np.abs(y_preds - target_lower)
    if only_valid == False:
        half_PI_length = np.mean(half_PI_lengths)
    else:
        valid_list = []
        for i in range(len(y_preds)):
            if y_trues[i] > lowers[target_error_rate][i] and y_trues[i] < uppers[target_error_rate][i]:
                valid_list.append(half_PI_lengths[i])
        half_PI_length = np.mean(valid_list)
    return half_PI_length


def x_percentage_coverage(x, y_preds, y_trues, lowers, uppers):
    alpha = round(1-x, 4)
    in_range_count = 0
    for i in range(len(y_preds)):
        if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
            in_range_count += 1
    cov_rate = in_range_count / len(y_preds)
    return cov_rate


def QICE(y_preds, lowers, n_bins=5):
    def intvl_counts(data, n_intvls, intvl_lb, intvl_ub):
        # get intervals
        intvls = np.linspace(intvl_lb, intvl_ub, num=n_intvls+1)
        # count number
        counts = np.zeros(n_intvls)
        for i in range(n_intvls):
            lb = intvls[i]
            ub = intvls[i+1]
            for k in range(len(data)):
                if data[k] <= ub and data[k] >= lb:
                    counts[i] += 1
        return counts
    alphas = list(lowers.keys())
    exp_counts = intvl_counts(alphas, n_intvls=n_bins, intvl_lb=0, intvl_ub=1)
    exp_counts = np.flip(exp_counts)
    qice_scores = []
    for i in range(len(y_preds)):
        cur_qs = [np.abs(y_preds[i] - lowers[alpha][i]) for alpha in lowers]
        total_counts = len(cur_qs)
        cur_counts = intvl_counts(data=cur_qs, n_intvls=n_bins, intvl_lb=np.min(cur_qs), intvl_ub=np.max(cur_qs))
        qice_score = np.mean(np.abs(cur_counts/total_counts - exp_counts/total_counts)) 
        qice_scores.append(qice_score)
    return np.mean(qice_scores)
        

def eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False, valid_PIL=False):
    """_summary_

    Args:
        y_preds (list): []
        y_trues (list): []
        lowers (dict): alpha -> []
        uppers (dict): alpha -> []
    """
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)

    # weighted IS
    weighted_IS_vals = weighted_IS(y_preds, y_trues, lowers.copy())
    mean_wis = np.mean(weighted_IS_vals)
    
    # crps
    stds = calculate_stds(y_preds, lowers.copy())
    crps_val = crps(y_preds, stds, y_trues)
    
    norm_crps_val = norm_crps(y_preds, y_trues, lowers.copy())
    
    rmse_val = rmse(y_preds, y_trues)
    nrmse_val = norm_rmse(y_preds, y_trues)
    mape_val = mape(y_preds, y_trues)
    # cs score
    cs_score = 0
    for alpha in lowers:
        in_range_count = 0
        for i in range(len(y_preds)):
            if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
                in_range_count += 1
        cov_rate = in_range_count / len(y_preds)
        cs_score += 1 / len(lowers) * abs(cov_rate - 1 + alpha)
    # monotonicity
    mono_score = monotonicity(y_preds, lowers)
    
    # validity
    dist_valid_score = validity_score(y_preds, lowers)
    
    # 90% PI half length
    pi_length_90 = interval_length(y_preds, lowers, target_error_rate=0.1) if not valid_PIL else interval_length(y_preds, lowers, target_error_rate=0.1, only_valid=True, y_trues=y_trues, uppers=uppers)
    
    # 50% coverage
    cov_90 = x_percentage_coverage(0.90, y_preds, y_trues, lowers, uppers)
    
    # 50% coverage
    cov_50 = x_percentage_coverage(0.5, y_preds, y_trues, lowers, uppers)
    
    # 95% coverage
    cov_95 = x_percentage_coverage(0.95, y_preds, y_trues, lowers, uppers)
    
    # QICE
    qice_score = QICE(y_preds, lowers, n_bins=5)
    
    if verbose:
        print(rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis, mono_score, cov_50, cov_95, qice_score)
    return rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis, mono_score, cov_50, cov_95, cov_90, pi_length_90, dist_valid_score, qice_score