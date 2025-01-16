import os
import time
import numpy as np
import yaml
from pathlib import Path
import copy
import pickle
import argparse

from forecaster.e2ecp import load_all_params
from forecaster.utils import pickle_save, pickle_load
from forecaster.eval_cp import prepare_data, eval_metrics
from hyperopt import fmin, tpe, hp, space_eval, Trials


# convert a saved trial object to acceptable format for hyperopt
def load_trials(trials_file_name, space):
    trials = pickle.load(open(trials_file_name, "rb"))
    best_trial = {}
    for key, val in trials.best_trial['misc']['vals']:
        best_trial[key] = val[0]
    print(space_eval(space, best_trial))


def write_yaml(fname, data):
    with open(fname, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def update_params(base_params:list, params_list):
    new_params = copy.deepcopy(base_params)
    for key, val in params_list.items():
        new_params[key] = val
    return new_params


def eval_one_exp(exp_name, alphas):    
    saved_pred_file = Path(f'../../results/{exp_name}.pkl')
    preds_in_right_format, regions, week_aheads = prepare_data(saved_pred_file, alphas)
    metrics = {}
    for region in regions:
        metrics[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
            # print(f'### Region is {region}, week ahead is {week_ahead}.')
            metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False)
    return metrics


def aggregrate_cs(exp_ids, alphas, weights=[80, 10, 10, 0]):
    cs_vals = []
    crps_vals = []
    wis_vals = []
    pil_vals = []
    for exp_id in exp_ids:
        metrics = eval_one_exp(exp_id, alphas)
        # get mean
        regions = list(metrics.keys())
        step_aheads = list(metrics[regions[0]].keys())
        for region in regions:
            for step_ahead in step_aheads:
                rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis, mono_score, cov_50, cov_95, cov_90, pi_length_90, dvs = metrics[region][step_ahead]
                cs_vals.append(cs_score)
                crps_vals.append(norm_crps_val)
                wis_vals.append(mean_wis)
                pil_vals.append(pi_length_90)
    return np.nanmean(cs_vals) * weights[0] + np.nanmean(crps_vals) * weights[1] + np.nanmean(wis_vals) * weights[2] + np.nanmean(pil_vals) * weights[3]


def black_box_function(sampled_params):
    seeds = [1, 2, 3]
    exp_start_id = yaml_file_id # exclusive
    
    ##########################################
    # write yaml files using input arguments #
    ##########################################
    
    # clear pkl files from previous runs
    res_dir = Path('../../results')
    for i in range(len(seeds)):
        cur_path = res_dir / f'{exp_start_id + i + 1}.pkl'
        if cur_path.exists():
            cur_path.unlink()
    
    # load params
    e2ecp_params = load_all_params(exp_start_id)
    model_params = e2ecp_params['model_params']
    alphas = e2ecp_params['alphas']
    
    cur_exp_id = exp_start_id
    yaml_folder_path = Path('../../setup/cp_exp_params')
    
    # write params
    for seed in seeds:
        cur_params = copy.deepcopy(e2ecp_params)
        cur_exp_id += 1
        yaml_file_name = f'{cur_exp_id}.yaml'
        cur_params['seed'] = seed
        cur_params['model_params'] = update_params(model_params, sampled_params)
        
        # manully add
        # cur_params['model_params']['informer_shared_params']['informer_d_model'] = sampled_params['informer_d_model']
        # cur_params['training_params']['K'] = sampled_params['K']
        # cur_params['training_params']['loss_factors'][2][2] = sampled_params['eloss_factor']
        # cur_params['training_params']['loss_factors'][2][0] = sampled_params['qloss_factor']
        # cur_params['training_params']['retrain_epochs'][-1] = sampled_params['third_stage_epoch']
        # cur_params['training_params']['cov_window_size'] = sampled_params['cov_window_size']
        print(cur_params)
        write_yaml(yaml_folder_path/yaml_file_name, cur_params)
    
    # submit jobs
    for i in range(len(seeds)):
        exp_id = exp_start_id + i + 1
        os.system(f'sbatch ../../bin/gl_job.sh {exp_id}')
    
    # wait
    success_run = False
    total_wait_seconds = 20000
    delta_time = 10
    current_wait_time = 0
    while(current_wait_time < total_wait_seconds):
        ready = True
        for i in range(len(seeds)):
            exp_id = exp_start_id + i + 1
            cur_path = res_dir / f'{exp_id}.pkl'
            if not cur_path.exists():
                ready = False
                break
        if ready:
            success_run = True
            break
        else:
            time.sleep(delta_time)
    
    # aggregrate results
    cs_score = 1
    if success_run:
        exp_ids = [exp_start_id + i + 1 for i in range(len(seeds))]
        cs_score = aggregrate_cs(exp_ids=exp_ids, alphas=alphas)
    return cs_score


# params_space = {
#     'hidden_dim': hp.choice('hidden_dim', [32, 64, 128, 256]),
#     'error_encoder_hidden_dim': hp.choice('error_encoder_hidden_dim', [8, 16, 64]),
#     'qhat_encoder_hidden_dim': hp.choice('qhat_encoder_hidden_dim', [32, 64, 128]),
#     'qhat_encoder_num_layers': hp.choice('qhat_encoder_num_layers', [2, 4, 8]),
    
#     'qhat_encoder_num_heads': hp.choice('qhat_encoder_num_heads', [1, 4, 16]),
#     'qhat_encoder_rd_hidden_dim': hp.choice('qhat_encoder_rd_hidden_dim', [32, 64, 128]),
#     'qhat_encoder_rd_num_layers': hp.choice('qhat_encoder_rd_num_layers', [2, 4, 8]),
#     'alpha_encoder_hidden_dim': hp.choice('alpha_encoder_hidden_dim', [32, 64, 128]),
    
#     'score_encoder_hidden_dim': hp.choice('score_encoder_hidden_dim', [32, 64, 128, 256]),
#     'score_encoder_num_layers': hp.choice('score_encoder_num_layers', [2, 4, 8]),
#     'q_fc_hidden_dim': hp.choice('q_fc_hidden_dim', [32, 64, 128, 256]),
    
#     # add to params manully
#     # model params -> informer shared params
#     'informer_d_model': hp.choice('informer_d_model', [32, 64, 128, 256, 512]),
#     # training params
#     'K': hp.loguniform('K', -3, 3),
#     'eloss_factor': hp.loguniform('eloss_factor', -10, 0),
# }

params_space = {
    # 'hidden_dim': hp.choice('hidden_dim', [32, 256]),
    # add to params manully
    # model params -> informer shared params
    # 'informer_d_model': hp.choice('informer_d_model', [32, 64, 128, 256, 512]),
    # training params
    # 'K': hp.loguniform('K', -3, 3),
    # 'eloss_factor': hp.loguniform('eloss_factor', -10, 0),
    # 'qloss_factor': hp.loguniform('qloss_factor', -10, 2),
    # 'third_stage_epoch': hp.choice('third_stage_epoch', [1, 5, 9]),
    # 'cov_window_size': hp.choice('cov_window_size', [5, 10, 15, 20, 25]),
    'encoder_hidden_dim': hp.choice('encoder_hidden_dim', [16, 64, 256, 512]),
    'encoder_num_layers': hp.choice('encoder_num_layers', [2, 4, 8, 16]),
    'encoder_heads': hp.choice('encoder_heads', [2, 4, 8, 16]),
    'attn1_num_heads': hp.choice('attn1_num_heads', [2, 4, 8, 16]),
}


def run_hyper_opt(trials_file:str, results_file:str, rounds:int):
    trials_path = Path(trials_file)
    if trials_path.exists():
            best_param = fmin(
                fn=black_box_function,
                space=params_space,
                max_evals=rounds,
                algo=tpe.suggest,
                trials_save_file=trials_path,
            )
    else:
        trials = Trials()
        best_param = fmin(
            fn=black_box_function,
            space=params_space,
            max_evals=rounds,
            algo=tpe.suggest,
            trials=trials,
            trials_save_file=trials_path,
        )
    best_params_to_save = space_eval(params_space, best_param)
    pickle_save(results_file, best_params_to_save)


yaml_file_id = 30001

# example run: python bayesian_tuning.py -r -e=1 -i=2 -y=1000
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', action='store_true')
    parser.add_argument('--exp_id', '-e')
    parser.add_argument('--iters', '-i')
    parser.add_argument('--yaml_file', '-y')
    parser.add_argument('--read_trial', '-t', action='store_true')
    args = parser.parse_args()
    exp_id = int(args.exp_id)
    yaml_file_id = int(args.yaml_file)
    iters = int(args.iters)
    trials_file = f'../../results/hyperopt_{exp_id}.trials'
    results_file = f'../../results/hyperopt_{exp_id}.pkl'
    if args.run:
        run_hyper_opt(trials_file=trials_file, results_file=results_file, rounds=iters)
    elif args.read_trial:
        trials = pickle_load(trials_file, version5=True)
        best_trial = {}
        for key, val in trials.best_trial['misc']['vals'].items():
            best_trial[key] = val[0]
        print(space_eval(params_space, best_trial))
    else:
        best_param = pickle_load(results_file, version5=True)
        print(best_param)
