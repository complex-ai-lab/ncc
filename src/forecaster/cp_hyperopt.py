import os
import time
import numpy as np
import yaml
from pathlib import Path
import copy
import pickle
import argparse
from hyperopt import fmin, tpe, hp, space_eval, Trials

from forecaster.online_training import get_params
from forecaster.e2ecp import load_all_params
from forecaster.utils import pickle_save, pickle_load
from forecaster.cpmethods import prepare_scores, aci, quantile_integrator_log_scorecaster
from forecaster.metrics import eval_metrics


# convert a saved trial object to acceptable format for hyperopt
def load_trials(trials_file_name, space):
    trials = pickle.load(open(trials_file_name, "rb"))
    best_trial = {}
    for key, val in trials.best_trial['misc']['vals']:
        best_trial[key] = val[0]
    print(space_eval(space, best_trial))


def get_uncertainty_est(base_pred, alphas, regions, aheads, method='aci', total_steps=-1, params=None):
    """
    If total_steps is not -1, take the last total_steps.
    """
    # prepare data and scores (use saved_pred)
    
    preds_in_right_format = {}
    pred_total_steps = None
    skip_beginning = 0
    
    for region in regions:
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


def get_score_from_pred(preds, weights=[10, 1]):
    # average cs and crps
    cs_vals = []
    crps_vals = []
    wis_vals = []
    for region in regions:
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds[region][week_ahead]
            rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis, mono_score, cov_50, cov_95, cov_90, pi_length_90, dvs = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False)
            cs_vals.append(cs_score)
            crps_vals.append(norm_crps_val)
            wis_vals.append(mean_wis)
    score = np.nanmean(cs_vals) * weights[0] + np.nanmean(pi_length_90) * weights[1]
    return score


def aci_score_func(sampled_params):
    preds = get_uncertainty_est(base_pred, alphas, regions, week_aheads, 'aci', params=sampled_params)
    score = get_score_from_pred(preds)
    return score


def pid_score_func(sampled_params):
    preds = get_uncertainty_est(base_pred, alphas, regions, week_aheads, 'pid', params=sampled_params)
    score = get_score_from_pred(preds)
    return score


def get_hyperopt_params(method):
    aci_param_space = {
        'lr': hp.loguniform('lr', -3, 0),
        'window_length': hp.choice('window_length', [5, 10, 15, 20, 25, 30]),
    }
    pid_param_space = {
        'lr': hp.loguniform('lr', -5, 1),
        'Csat': hp.loguniform('Csat', -2, 8),
        'KI': hp.loguniform('KI', -8, 2),
    }
    if method == 'aci':
        return aci_score_func, aci_param_space
    if method == 'pid':
        return pid_score_func, pid_param_space


def run_hyper_opt(trials_file:str, results_file:str, rounds:int, method):
    # TODO: run without using trials_save_file is fine. When use the trials_save_file, exception is throwed.
    score_func, params_space = get_hyperopt_params(method)
    trials_path = Path(trials_file)
    if trials_path.exists():
            best_param = fmin(
                fn=score_func,
                space=params_space,
                max_evals=rounds,
                algo=tpe.suggest,
                trials_save_file=trials_path,
            )
    else:
        trials = Trials()
        best_param = fmin(
            fn=score_func,
            space=params_space,
            max_evals=rounds,
            algo=tpe.suggest,
            trials=trials,
            # trials_save_file=trials_path,
        )
    best_params_to_save = space_eval(params_space, best_param)
    pickle_save(results_file, best_params_to_save)


# TODO: change cp input and base input for different tasks
# flu_hosp: 0, 4, covid: 100, 20, power: 200, 11
cp_input = 610
base_input = 50

e2ecp_params = load_all_params(cp_input)
base_pred_params = get_params(base_input)

data_id = e2ecp_params['data_file_id']

# global variables
alphas = e2ecp_params['alphas']
week_aheads = e2ecp_params['week_ahead_list']
regions = base_pred_params['regions']
base_pred = pickle_load(f'../../results/base_pred/saved_pred_{data_id}.pickle', version5=True)['base_pred']


# example run: python bayesian_tuning.py -r -e=1 -i=2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', action='store_true')
    parser.add_argument('--exp_id', '-e')
    parser.add_argument('--iters', '-i')
    parser.add_argument('--method', '-m')
    args = parser.parse_args()
    exp_id = int(args.exp_id)
    iters = int(args.iters)
    method = str(args.method)
    trials_file = f'../../results/hyperopt_{exp_id}.trials'
    results_file = f'../../results/hyperopt_{exp_id}.pkl'
    if args.run:
        run_hyper_opt(trials_file=trials_file, results_file=results_file, rounds=iters, method=method)
    else:
        best_param = pickle_load(results_file, version5=True)
        print(best_param)
