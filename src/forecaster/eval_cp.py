from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from epiweeks import Week
import yaml
import argparse
from sklearn import linear_model
import csv

from forecaster.data_utils import epiweek_sub, convert_to_epiweek
from forecaster.utils import pickle_load, pickle_save, load_yaml_params
from forecaster.metrics import eval_metrics
from forecaster.eval import WriteRows2CSV
from forecaster.cpmethods import get_uncertainty_est
from forecaster.e2ecp import load_all_params
from forecaster.visualize_helper import *


metric_names = ['rmse', 'nrmse', 'mape', 'cs_score', 'norm_crps_val', 'mean_wis', 'mono', 'cov_50', 'cov_95', 'cov_90', 'pi_length_90', 'dvs', 'qice']


def get_meta_info(predictions):
    regions = list(predictions.keys())
    week_aheads = list(predictions[regions[0]].keys())
    return regions, week_aheads


def prepare_data(saved_pred_file, alphas):
    """qhats are converted to zero is less than zero. qhat is set to zero if alpha=1"""
    predictions = pickle_load(saved_pred_file, version5=True)
    regions, week_aheads = get_meta_info(predictions)
    preds_in_right_format = {}
    for region in regions:
        preds_in_right_format[region] = {}
        for week_ahead in week_aheads:
            y_trues = []
            y_preds = []
            lowers = {}
            uppers = {}
            for alpha in alphas:
                lowers[alpha] = []
                uppers[alpha] = []
            cur_predictions = predictions[region][week_ahead]
            for weekly_preds in cur_predictions:
                q_hats, y_hat, y = weekly_preds
                q_hats = np.array(q_hats)
                q_hats[q_hats < 0] = 0
                y_val = y[0]
                y_hat_val = y_hat[0]
                y_trues.append(y_val)
                y_preds.append(y_hat_val)
                for i, alpha in enumerate(alphas):
                    # if alpha == 1:
                    #     lowers[alpha].append(y_hat_val)
                    #     uppers[alpha].append(y_hat_val)
                    # else:
                        # lowers[alpha].append(y_hat[0, 0] - max(0, q_hats[0, i]))
                        # uppers[alpha].append(y_hat[0, 0] + max(0, q_hats[0, i]))
                    lowers[alpha].append(y_hat_val - q_hats[i])
                    uppers[alpha].append(y_hat_val + q_hats[i])
            preds_in_right_format[region][week_ahead] = (y_trues, y_preds, lowers, uppers)
    return preds_in_right_format, regions, week_aheads


def save_metrics2csv(metrics, path):
    rows = []
    header = ['region', 'week ahead', ] + metric_names
    rows.append(header)
    region_rows = []
    regions = list(metrics.keys())
    for region in regions:
        for week_ahead in metrics[region]:
            cur_row = [region, week_ahead]
            metric = metrics[region][week_ahead]
            for i in range(len(metric)):
                cur_row.append(metric[i])
            region_rows.append(cur_row)
    # median
    for w in metrics[regions[0]]:
        tmp_dict = {i: [] for i in range(len(region_rows[0])-2)}
        for row in region_rows:
            if row[1] == w:
                for idx in tmp_dict:
                    tmp_dict[idx].append(row[idx+2])
        cur_row = ['median', w]
        for _, tmp_list in tmp_dict.items():
            cur_row.append(np.median(tmp_list))
        rows.append(cur_row)
    
    # mean
    for w in metrics[regions[0]]:
        tmp_dict = {i: [] for i in range(len(region_rows[0])-2)}
        for row in region_rows:
            if row[1] == w:
                for idx in tmp_dict:
                    tmp_dict[idx].append(row[idx+2])
        cur_row = ['mean', w]
        for _, tmp_list in tmp_dict.items():
            cur_row.append(np.mean(tmp_list))
        rows.append(cur_row)
    rows += region_rows
    WriteRows2CSV(rows, path)


def clip_ts_with_index(preds_in_right_format, regions, week_aheads, start_idx, end_idx):
    clipped_preds = {}
    for region in regions:
        clipped_preds[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
            alphas = list(lowers.keys())
            clipped_lowers = {}
            clipped_uppers = {}
            for alpha in alphas:
                clipped_lowers[alpha] = lowers[alpha][start_idx:end_idx]
                clipped_uppers[alpha] = uppers[alpha][start_idx:end_idx]
            clipped_preds[region][week_ahead] = (
                y_trues[start_idx:end_idx],
                y_preds[start_idx:end_idx],
                clipped_lowers,
                clipped_uppers,
            )
    return clipped_preds


def list_idx(mylist, myidx):
    return [mylist[i] for i in myidx]


def clip_ts_with_index_list(preds_in_right_format, regions, week_aheads, idxes):
    clipped_preds = {}
    for region in regions:
        clipped_preds[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
            alphas = list(lowers.keys())
            clipped_lowers = {}
            clipped_uppers = {}
            for alpha in alphas:
                clipped_lowers[alpha] = list_idx(lowers[alpha], idxes)
                clipped_uppers[alpha] = list_idx(uppers[alpha], idxes)
            clipped_preds[region][week_ahead] = (
                list_idx(y_trues, idxes),
                list_idx(y_preds, idxes),
                clipped_lowers,
                clipped_uppers,
            )
    return clipped_preds


def clip_week_range(preds_in_right_format, regions, week_aheads, start_week, eval_start_week, eval_end_week, from_end=False, weeks_from_end=0):
    start_idx = epiweek_sub(eval_start_week, start_week)
    end_idx = epiweek_sub(eval_end_week, start_week)
    total_weeks = len(preds_in_right_format[regions[0]][week_aheads[0]][0])
    weeks_from_end_in_e2ecp = total_weeks - end_idx
    total_eval_weeks = epiweek_sub(convert_to_epiweek(eval_end_week), convert_to_epiweek(eval_start_week))
    if from_end:
        print(f'weeks_from_end: {weeks_from_end}')
        print(f'total_eval_weeks: {total_eval_weeks}')
        start_idx = total_weeks - total_eval_weeks - weeks_from_end
        end_idx = total_weeks - weeks_from_end
    clipped_preds = {}
    for region in regions:
        clipped_preds[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
            alphas = list(lowers.keys())
            clipped_lowers = {}
            clipped_uppers = {}
            for alpha in alphas:
                clipped_lowers[alpha] = lowers[alpha][start_idx:end_idx]
                clipped_uppers[alpha] = uppers[alpha][start_idx:end_idx]
            clipped_preds[region][week_ahead] = (
                y_trues[start_idx:end_idx],
                y_preds[start_idx:end_idx],
                clipped_lowers,
                clipped_uppers,
            )
    return clipped_preds, weeks_from_end_in_e2ecp


def sort_preds(preds, regions, week_aheads):
    sorted_preds = {}
    for region in regions:
        sorted_preds[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds[region][week_ahead]
            alphas = list(lowers.keys())
            sorted_lowers = {}
            sorted_uppers = {}
            for alpha in alphas:
                sorted_lowers[alpha] = []
                sorted_uppers[alpha] = []
            for i in range(len(y_preds)):
                halfPIs = {} # half PIs in one time step
                for alpha in alphas:
                    halfPIs[alpha] = y_preds[i] - lowers[alpha][i]
                sorted_keys = sorted(list(halfPIs.keys()))
                sorted_vals = sorted(list(halfPIs.values()), reverse=True)
                for j in range(len(sorted_keys)):
                    sorted_lowers[sorted_keys[j]].append(y_preds[i] - sorted_vals[j])
                    sorted_uppers[sorted_keys[j]].append(y_preds[i] + sorted_vals[j])
            sorted_preds[region][week_ahead] = (
                y_trues,
                y_preds,
                sorted_lowers,
                sorted_uppers,
            )
    return sorted_preds


def eval_one_exp(e2ecp_params, exp_name, clip_idx_list=None, sort_res=False):    
    saved_pred_file = Path(f'../../results/{exp_name}.pkl')
    alphas = e2ecp_params['alphas']
    preds_in_right_format, regions, week_aheads = prepare_data(saved_pred_file, alphas)
    if sort_res:
        preds_in_right_format = sort_preds(preds_in_right_format, regions, week_aheads)
    if clip_idx_list is not None:
        preds_in_right_format = clip_ts_with_index_list(preds_in_right_format, regions, week_aheads, clip_idx_list)
    metrics = {}
    for region in regions:
        metrics[region] = {}
        for week_ahead in week_aheads:
            (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
            # print(f'### Region is {region}, week ahead is {week_ahead}.')
            metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False)
    return metrics


def aggregrate_results_across_exps(input_id, clip_idx_list=None, sort_res=False, save_dir='../../results/csvs'):
    e2ecp_params = load_all_params(input_id)
    header = ['seed', 'week ahead', ] + metric_names
    rows = []
    rows.append(header)
    exp_rows = []
    for seed in e2ecp_params['seeds']:
        seed_success = True
        try:
            exp_name = f'{input_id}_{seed}'
            metrics = eval_one_exp(e2ecp_params, exp_name, clip_idx_list=clip_idx_list, sort_res=sort_res)
            # get mean for each step ahead
            regions = list(metrics.keys())
            step_aheads = list(metrics[regions[0]].keys())
            region_rows = []
            for region in regions:
                for step_ahead in step_aheads:
                    cur_row = [region, step_ahead]
                    metric = metrics[region][step_ahead]
                    for i in range(len(metric)):
                        cur_row.append(metric[i])
                    region_rows.append(cur_row)
        except:
            seed_success = False
            print(f'Seed {seed} does not exist!')
        
        # mean of regions
        if seed_success:
            for w in metrics[regions[0]]:
                tmp_dict = {i: [] for i in range(len(region_rows[0])-2)}
                for row in region_rows:
                    if row[1] == w:
                        for idx in tmp_dict:
                            tmp_dict[idx].append(row[idx+2])
                cur_row = [f'{seed}', w]
                for _, tmp_list in tmp_dict.items():
                    cur_row.append(np.mean(tmp_list))
                exp_rows.append(cur_row)
    # mean of seeds
    for w in metrics[regions[0]]:
        tmp_dict = {i: [] for i in range(len(exp_rows[0])-2)}
        for row in exp_rows:
            if row[1] == w:
                for idx in tmp_dict:
                    tmp_dict[idx].append(row[idx+2])
        cur_row = ['mean', w]
        for _, tmp_list in tmp_dict.items():
            cur_row.append(np.mean(tmp_list))
        rows.append(cur_row)
    rows += exp_rows
    WriteRows2CSV(rows, f'{save_dir}/{input_id}_multi_seeds.csv')


def clip_index_prep(base_pred_ds, e2ecp_params):
    clip_idx_list = None
    if base_pred_ds == 'SKIPcovid':
        clip_idx_file_path = Path(f'../../data/{base_pred_ds}_{e2ecp_params["data_file_id"]}_clip_idx.pkl')
    elif base_pred_ds == 'covid' and e2ecp_params["data_file_id"] == 4:
        clip_idx_file_path = Path(f'../../data/covid_4_clip_idx.pkl')
    else:
        clip_idx_file_path = Path(f'../../data/{base_pred_ds}_clip_idx.pkl')
    if clip_idx_file_path.exists():
        clip_idx_list = pickle_load(clip_idx_file_path, version5=True)
    else:
        print('clip index file not found')
    return clip_idx_list


def get_preds(methods, params, exp_id, seed, base_pred_ds, clip_idx_list, sort):
    # must start with our method
    input_id = f'{exp_id}_{seed}'
    test_weeks = int(params['training_params']['test_weeks'])
    data_id = params['data_file_id']
    alphas = params['alphas']
    
    all_preds = {}
    
    regions = None
    week_aheads = None
    for method in methods:
        cur_pred = None
        if method == 'ncc':
            saved_pred_file = Path(f'../../results/{input_id}.pkl')
            cur_pred, regions, week_aheads = prepare_data(saved_pred_file, alphas)
        else:
            cp_param_opt = 'elec' if base_pred_ds == 'electricity' else base_pred_ds
            cp_param = cp_params[cp_param_opt]
            # print(weeks_from_end)
            if method == 'cfrnn' or method == 'nexcp' or method == 'faci':
                cp_method_params = cp_param['aci']
            else:
                cp_method_params = cp_param[method]
            cur_pred = get_uncertainty_est(data_id=data_id, alphas=alphas, regions=regions, aheads=week_aheads, method=method, total_steps=test_weeks, params=cp_method_params)
        if clip_idx_list is not None:
            cur_pred = clip_ts_with_index_list(cur_pred, regions, week_aheads, clip_idx_list)
        if sort:
            cur_pred = sort_preds(cur_pred, regions, week_aheads)
        all_preds[method] = cur_pred
    return all_preds
    

cp_params = {
    "flu": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 10,
            'KI': 1,
            'lr': 0.04,
        }
    },
    "covid": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 32,
            'KI':  0.6,
            'lr': 0.03,
        }
    },
    "power": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 0.83137,
            'KI':  27.79276,
            'lr': 0.32,
        }
    },
    "weather": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 0.83137,
            'KI':  27.79276,
            'lr': 0.32,
        }
    },
    "elec": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 227.5,
            'KI':  0.0945,
            'lr': 1.476,
        }
    },
    "smd": {
        'aci': {
            'lr': 0.05,
            'window_length': 15
        },
        'pid': {
            'Csat': 227.5,
            'KI':  0.0945,
            'lr': 1.476,
        }
    }
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    parser.add_argument('--seed', '-s', default=-1, help='When evaluation a multi-seed experiment, a seed needs to be specified.')
    parser.add_argument('--multi_seeds', '-u', action='store_true')
    parser.add_argument('--method', '-m', help="Method")
    parser.add_argument('--clip', '-c', action='store_true')
    parser.add_argument('--sort', '-o', action='store_true')
    parser.add_argument('--cp_param_opt', '-p')
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('-ov', '--only_valid_pil', action='store_true')
    args = parser.parse_args()
    visualize_opt = args.visualize
    input_file = args.input    # for example: 0
    input_id = int(input_file)
    input_id_wseed = f'{input_id}_{args.seed}' if args.seed != -1 else f'{input_id}'
    method = str(args.method)
    sort_cp_res = args.sort
    
    only_valid_pil = args.only_valid_pil
    
    # clip or not
    e2ecp_params = load_all_params(input_id)
    all_base_preds = pickle_load(f'../../results/base_pred/saved_pred_{e2ecp_params["data_file_id"]}.pickle', version5=True)
    base_pred_ds = all_base_preds['params']['dataset']
    clip_idx_list = None
    if args.clip:
        if base_pred_ds == 'SKIPcovid':
            clip_idx_file_path = Path(f'../../data/{base_pred_ds}_{e2ecp_params["data_file_id"]}_clip_idx.pkl')
        elif base_pred_ds == 'covid' and e2ecp_params["data_file_id"] == 4:
            clip_idx_file_path = Path(f'../../data/covid_4_clip_idx.pkl')
        else:
            clip_idx_file_path = Path(f'../../data/{base_pred_ds}_clip_idx.pkl')
        if clip_idx_file_path.exists():
            clip_idx_list = pickle_load(clip_idx_file_path, version5=True)
        else:
            print('clip index file not found')
    
    if args.multi_seeds:
        aggregrate_results_across_exps(input_id, clip_idx_list, sort_res=sort_cp_res)
    else:
        # get start week
        e2ecp_params = load_all_params(input_id)
        
        data_id = e2ecp_params['data_file_id']
        val_weeks = int(e2ecp_params['training_params']['val_weeks'])
        test_weeks = int(e2ecp_params['training_params']['test_weeks'])
        print(f'number of test weeks is {test_weeks}')
        print(data_id)
        
        saved_pred_file = Path(f'../../results/{input_id_wseed}.pkl')
        alphas = e2ecp_params['alphas']
        preds_in_right_format, regions, week_aheads = prepare_data(saved_pred_file, alphas)
        
        if method != 'e2ecp':
            cp_param_opt = args.cp_param_opt
            if cp_param_opt is None:
                cp_param_opt = 'elec' if base_pred_ds == 'electricity' else base_pred_ds
            cp_param = cp_params[cp_param_opt]
            # print(weeks_from_end)
            if method == 'cfrnn' or method == 'nexcp' or method == 'faci':
                cp_method_params = cp_param['aci']
            else:
                cp_method_params = cp_param[method]
            preds_in_right_format = get_uncertainty_est(data_id=data_id, alphas=alphas, regions=regions, aheads=week_aheads, method=method, total_steps=test_weeks, params=cp_method_params)
        
        
        if clip_idx_list is not None:
            preds_in_right_format = clip_ts_with_index_list(preds_in_right_format, regions, week_aheads, clip_idx_list)
        
        if sort_cp_res:
            preds_in_right_format = sort_preds(preds_in_right_format, regions, week_aheads)
        
        # plot_ground_truth(data_id, 'US', 0, 0, -1)
        
        if visualize_opt:
            try:
                plot_predictions_with_cis(preds_in_right_format, 'X', 1, [0.3, 0.5, 0.7], fill=True, max_steps=200)
            except:
                plot_predictions_with_cis(preds_in_right_format, 'US', 1, [0.3, 0.5, 0.7], fill=True, max_steps=200)
        
        
        SVAE_RUNNING_CS = False
        SAVE_PIL = False
        VIZ_COV = True
        
        if SVAE_RUNNING_CS:
            save_running_cs_data(preds_in_right_format, 'US', 1, 15, f'../../results/{input_id}_{method}_avg_cs.npy', cumsum=True)
        if SAVE_PIL:
            save_pil_data(preds_in_right_format, 'X', 1, 0.1, f'../../results/{input_id}_{method}_90pil.pkl')
        if VIZ_COV:
            plot_cov_pred_in_one_figure(preds_in_right_format=preds_in_right_format, region='US', week_ahead=4, alpha=0.1, window_size=10, max_steps=40, save_path='../../results/tmp/new_plot.pdf')
            
        
        # cov_plot_dir = Path(f'../../results/cov_{method}_{input_id}')
        # if not cov_plot_dir.exists():
        #     cov_plot_dir.mkdir()
        # try:
        #     plot_avg_cov_curve(preds_in_right_format, 'US', 1, 0.1, 15, 100, True, figpath=cov_plot_dir/f'cov.jpg')
        # except:
        #     plot_avg_cov_curve(preds_in_right_format, 'X', 1, 0.1, 15, 100, True, figpath=cov_plot_dir/f'cov.jpg')
        
        
        # saved_losses_file = Path(f'../../models/losses_{input_id}.pkl')
        # plot_losses(saved_losses_file, loss_types=['qloss', 'closs', 'eloss'], plot_train_loss=True, plot_val_loss=False)
        
        # plot_horizon_predictions(preds_in_right_format, 'US', [0.1, 0.3, 0.5], 20)
        
        # save_fig_folder = Path(f'../../results/{input_id}')
        # if not save_fig_folder.exists():
        #     save_fig_folder.mkdir()
        # for region in regions:
        #     plot_predictions_with_cis(preds_in_right_format, region, 1, [0.1, 0.3, 0.5, 0.7, 0.9], fill=True, savefig=True, figpath=save_fig_folder/f'{region}_{method}.png')
        
        # save_fig_folder = Path(f'../../results/{input_id}_horizon')
        # if not save_fig_folder.exists():
        #     save_fig_folder.mkdir()
        # num_weeks = epiweek_sub(convert_to_epiweek(eval_end_week), convert_to_epiweek(eval_start_week))
        # prev_weeks = 3
        # for region in regions:
        #     for week_idx_ in range(num_weeks-prev_weeks):
        #         week_idx = week_idx_ + prev_weeks
        #         plot_horizon_predictions(preds_in_right_format, region, [0.1, 0.3, 0.5], week_idx, prev_weeks=prev_weeks, savefig=True, figpath=save_fig_folder/f'{region}_{week_idx}.png')
        
        metrics = {}
        for region in regions:
            metrics[region] = {}
            for week_ahead in week_aheads:
                (y_trues, y_preds, lowers, uppers) = preds_in_right_format[region][week_ahead]
                # print(f'### Region is {region}, week ahead is {week_ahead}.')
                metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False, valid_PIL=only_valid_pil)
        
        # box_plot(1, 3, metrics)
        
        # # cs plots
        # cs_plot_dir = Path(f'../../results/cs_curve_{method}_{input_id}')
        # if not cs_plot_dir.exists():
        #     cs_plot_dir.mkdir()
        # for region in regions:
        #     for week_ahead in week_aheads:
        #         cs_curve(week_ahead, region, preds_in_right_format, save_path=cs_plot_dir / f'{region}_{week_ahead}.pdf')
                
        if method != 'e2ecp':
            save_metrics2csv(metrics, f'../../results/csvs/{input_id}_{method}.csv')
        else:
            save_metrics2csv(metrics, f'../../results/csvs/{input_id}.csv')
        
        if e2ecp_params['training_params']['region_fine_tuning']:
            saved_rft_pred_file = Path(f'../../results/{input_id}_rfted.pkl')
            rft_preds_in_right_format, regions, week_aheads = prepare_data(saved_rft_pred_file, alphas)
            # rft_preds_in_right_format = clip_week_range(rft_preds_in_right_format, regions, week_aheads, start_week, convert_to_epiweek(eval_start_week), convert_to_epiweek(eval_end_week))
            metrics = {}
            for region in regions:
                metrics[region] = {}
                for week_ahead in week_aheads:
                    (y_trues, y_preds, lowers, uppers) = rft_preds_in_right_format[region][week_ahead]
                    # print(f'### Region is {region}, week ahead is {week_ahead}.')
                    metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False)
                save_metrics2csv(metrics, f'../../results/csvs/{input_id}_{method}_rft.csv')



if __name__ == '__main__':    
    main()