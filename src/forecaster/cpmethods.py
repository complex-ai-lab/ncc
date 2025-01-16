import math
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
from scipy.special import logsumexp

from forecaster.utils import pickle_load


def nex_quantile(arr, q, gamma):
    q = np.clip(q, 0, 1)
    if len(arr) == 0:
        return np.zeros(len(q)) if hasattr(q, "__len__") else 0
    weights = np.exp(np.log(gamma) * np.arange(len(arr) - 1, -1, -1))
    assert len(weights) == len(arr)
    idx = np.argsort(arr)
    weights = np.cumsum(weights[idx])
    q_idx = np.searchsorted(weights / weights[-1], q)
    return np.asarray(arr)[idx[q_idx]]


def pinball_loss(y, yhat, q: float):
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))


def NexCP(scores,
    alpha,
    window_length,
    T_burnin,
    ahead,):
    
    coverage = 1 - alpha
    gamma = coverage + 3 * (1 - coverage) / 4
    
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
            qs[t] = nex_quantile(scores[max(t_pred-window_length,0):t_pred], np.clip(scale_factor*(1-alpha), 0, 1), gamma=gamma)
            covereds[t] = qs[t] >= scores[t]
        else:
            if t_pred > 0:
                scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
                qs[t] = nex_quantile(scores[:t_pred], np.clip(scale_factor*(1-alpha), 0, 1), gamma=gamma)
            else:
                qs[t] = np.max(scores)
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results



########
# FACI #
########


# Definition of the pinball loss function
def vec_zero_max(x):
    return np.maximum(x, 0)

def vec_zero_min(x):
    return np.minimum(x, 0)

def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

def find_beta(recent_scores, cur_score, epsilon=0.001):
    top = 1
    bot = 0
    mid = (top + bot) / 2
    
    while top - bot > epsilon:
        if np.quantile(recent_scores, 1 - mid) > cur_score:
            bot = mid
        else:
            top = mid
        mid = (top + bot) / 2
    
    return mid


# if __name__ == '__main__':
#     recent_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#     cur_score = 2
#     print(find_beta(recent_scores, cur_score))


def FACI(scores,
    window_length,
    T_burnin,
    alpha,
    gammas,
    sigma=1/1000,
    eta=2.72):
    betas = FACI_preprocess(scores, window_length, T_burnin)
    expert_results = conformal_adapt_stable(betas, alpha, gammas, sigma, eta)
    expert_alphas = expert_results['alpha_seq']
    
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        cur_alpha = expert_alphas[t]
        t_pred = t
        if t_pred > T_burnin:
            scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
            qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], np.clip(scale_factor*(1-cur_alpha), 0, 1))
            covereds[t] = qs[t] >= scores[t]

        else:
            if t_pred > 0:
                scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
                qs[t] = np.quantile(scores[:t_pred], np.clip(scale_factor*(1-cur_alpha), 0, 1))
            else:
                qs[t] = np.max(scores)
    results = { "method": "FACI", "q" : qs, "alpha" : expert_alphas}
    return results


def FACI_preprocess(scores,
    window_length,
    T_burnin,):
    betas = []
    for t in range(len(scores)):
        cur_beta = 0.01
        if t > T_burnin:  
            cur_score = scores[t]
            recent_scores = scores[max(0, t-window_length):t]
            cur_beta = find_beta(recent_scores, cur_score)
        betas.append(cur_beta)
    return betas


# FACI method implementation
def conformal_adapt_stable(betas, alpha, gammas, sigma=1/1000, eta=2.72):
    T = len(betas)
    k = len(gammas)
    
    alpha_seq = np.full(T, alpha)
    err_seq_adapt = np.zeros(T)
    err_seq_fixed = np.zeros(T)
    gamma_seq = np.zeros(T)
    mean_alpha_seq = np.zeros(T)
    mean_err_seq = np.zeros(T)
    mean_gammas = np.zeros(T)
    
    expert_alphas = np.full(k, alpha)
    expert_ws = np.ones(k)
    cur_expert = np.random.choice(k)
    expert_cumulative_losses = np.zeros(k)
    expert_probs = np.full(k, 1/k)
    
    for t in range(T):
        alphat = expert_alphas[cur_expert]
        alpha_seq[t] = alphat
        err_seq_adapt[t] = float(alphat > betas[t])
        err_seq_fixed[t] = float(alpha > betas[t])
        gamma_seq[t] = gammas[cur_expert]
        mean_alpha_seq[t] = np.dot(expert_probs, expert_alphas)
        mean_err_seq[t] = float(mean_alpha_seq[t] > betas[t])
        mean_gammas[t] = np.dot(expert_probs, gammas)
        
        expert_losses = pinball(betas[t] - expert_alphas, alpha)
        
        # Update expert alphas
        expert_alphas = expert_alphas + gammas * (alpha - (expert_alphas > betas[t]).astype(float))
        
        # Update expert weights
        if eta < np.inf:
            expert_bar_ws = expert_ws * np.exp(-eta * expert_losses)
            expert_next_ws = (1 - sigma) * expert_bar_ws / np.sum(expert_bar_ws) + sigma / k
            expert_probs = expert_next_ws / np.sum(expert_next_ws)
            cur_expert = np.random.choice(k, p=expert_probs)
            expert_ws = expert_next_ws
        else:
            expert_cumulative_losses += expert_losses
            cur_expert = np.argmin(expert_cumulative_losses)
    
    return {
        "alpha_seq": alpha_seq,
        "err_seq_adapt": err_seq_adapt,
        "err_seq_fixed": err_seq_fixed,
        "gamma_seq": gamma_seq,
        "mean_alpha_seq": mean_alpha_seq,
        "mean_err_seq": mean_err_seq,
        "mean_gammas": mean_gammas
    }


class ScoreScalar:
    def __init__(self) -> None:
        self.var = None
        self.mean = None
    
    def fit_transform(self, scores):
        self.var = np.var(scores)
        self.mean = np.mean(scores)
        nscores = (scores - self.mean) / self.var
        return nscores

    def inv_transform(self, nscores):
        scores = (nscores * self.var) + self.mean
        return scores


def prepare_scores(base_pred, target_region, ahead):
    scores = []
    y_preds = []
    y_trues = []
    ahead_idx = ahead-1
    for i in range(len(base_pred)):
        predictions, addition_infos = base_pred[i]
        y, _, _, _, _ = addition_infos[target_region]
        y_pred = predictions[target_region]
        y_trues.append(y[ahead_idx])
        y_preds.append(y_pred[ahead_idx])
        scores.append(np.abs(y_pred[ahead_idx] - y[ahead_idx]))
    scores = np.array(scores)
    return scores, y_preds, y_trues


def cf_rnn(
    scores,
    alpha,
    window_length,
    T_burnin,
    ahead,
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
            qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], np.clip(scale_factor*(1-alpha), 0, 1))
            covereds[t] = qs[t] >= scores[t]

            if t < T_test - 1:
                alphas[t+1] = alpha
        else:
            if t_pred > 0:
                scale_factor = 1 + 1 / (t_pred - max(t_pred-window_length,0) + 1)
                qs[t] = np.quantile(scores[:t_pred], np.clip(scale_factor*(1-alpha), 0, 1))
            else:
                qs[t] = np.max(scores)
    results = { "method": "CFRNN", "q" : qs, "alpha" : alphas}
    return results


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in range(T_test):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1))
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > 0:
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.max(scores)
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results


def mytan(x):
    if x >= np.pi/2:
        # print('aaa')
        return 100
    elif x <= -np.pi/2:
        # print('bbb')
        return -100
    else:
        return np.tan(x)


def saturation_fn_log(x, t, Csat, KI):
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    # if np.abs(tan_out) > 1e10:
    #     print(f'tan out is: {tan_out}, x is {x}')
    out = KI * tan_out
    return out


def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
    lr=0.05,
    Csat=30,
    KI=1,
):
    # Normalize scores
    scaler = ScoreScalar()
    scores = scaler.fit_transform(scores)
    
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))      
    seasonal_period = 1
    train_model = True
    
    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in range(T_test):
        t_lr = t
        t_lr_min = max(t_lr - 5, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        #if onesided_integrator:
        #    integrator_arg = np.clip(integrator_arg, 0, np.infty)
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        # Train and scorecast if necessary
        if scorecast and train_model and t_pred > 5 and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            pred = model.forecast(ahead).to_numpy()
            scorecasts[t+ahead] = pred[ahead-1]
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    
    # convert back
    qs = scaler.inv_transform(qs)
    return qs


def get_uncertainty_est(data_id, alphas, regions, aheads, method='aci', total_steps=-1, params=None):
    """
    If total_steps is not -1, take the last total_steps.
    """
    # setup params
    if params is None:
        params = {
            'Csat': 5,
            'lr': 0.008,
            'KI': 0.1,
            'window_length': 20
        }
    
    
    # prepare data and scores (use saved_pred)
    base_pred = pickle_load(f'../../results/base_pred/saved_pred_{data_id}.pickle', version5=True)['base_pred']
    
    preds_in_right_format = {}
    pred_total_steps = None
    skip_beginning = 0
    
    for region in tqdm(regions):
        preds_in_right_format[region] = {}
        for ahead in aheads:
            lowers = {}
            uppers = {}
            for alpha in alphas:
                lowers[alpha] = []
                uppers[alpha] = []
            scores, y_preds, y_trues = prepare_scores(base_pred, region, ahead)
            if pred_total_steps is None:
                pred_total_steps = len(y_preds)
                if total_steps != -1:
                    skip_beginning = pred_total_steps - total_steps
            for i, alpha in enumerate(alphas):
                qpreds = None
                if method == 'pid':
                    qpreds = quantile_integrator_log_scorecaster(
                        scores=scores,
                        alpha=alpha,
                        ahead=ahead,
                        scorecast=False,
                        integrate=True,
                        Csat=params['Csat'],
                        lr=params['lr'],
                        KI=params['KI']
                    )
                if method == 'aci':
                    qpreds = aci(
                        scores=scores,
                        alpha=alpha,
                        lr=params['lr'],
                        T_burnin=5,
                        window_length=params['window_length'],
                        ahead=ahead,
                    )['q']
                if method == 'cfrnn':
                    qpreds = cf_rnn(
                        scores=scores,
                        alpha=alpha,
                        T_burnin=5,
                        window_length=params['window_length'],
                        ahead=ahead,
                    )['q']
                if method == 'nexcp':
                    qpreds = NexCP(
                        scores=scores,
                        alpha=alpha,
                        T_burnin=5,
                        window_length=params['window_length'],
                        ahead=ahead,
                    )['q']
                if method == 'faci':
                    qpreds = FACI(
                        scores=scores,
                        window_length=params['window_length'],
                        T_burnin=5,
                        alpha=alpha,
                        gammas=np.asarray([0.001 * 2**k for k in range(8)]),
                    )['q']
                for j in range(len(scores) - skip_beginning):
                    idx = j + skip_beginning
                    y_pred = y_preds[idx]
                    lower = y_pred - qpreds[idx]
                    upper = y_pred + qpreds[idx]
                    lowers[alpha].append(lower)
                    uppers[alpha].append(upper)
            y_trues = y_trues[skip_beginning:]
            y_preds = y_preds[skip_beginning:]
            preds_in_right_format[region][ahead] = (y_trues, y_preds, lowers, uppers)
    return preds_in_right_format
    
    