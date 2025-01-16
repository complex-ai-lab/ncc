import torch
import numpy as np
import matplotlib.pyplot as plt
from forecaster.utils import pickle_load, pickle_save


def save_pil_data(preds_in_right_format, region, week_ahead, target_alpha, file_name):
    y_trues, y_preds, lowers, uppers = preds_in_right_format[region][week_ahead]
    data2save = {
        'y_true': y_trues,
        'y_pred': y_preds,
        'lower': lowers[target_alpha],
        'upper': uppers[target_alpha],
    }
    pickle_save(file_name, data2save)


def save_running_cs_data(preds_in_right_format, region, week_ahead, window_size, file_name, cumsum=True):
    y_trues, y_preds, lowers, uppers = preds_in_right_format[region][week_ahead]
    num_weeks = len(y_trues)
    cs4alphas = []
    for alpha in lowers:
        cur_upper = uppers[alpha]
        cur_lower = lowers[alpha]
        covs = []
        for i in range(num_weeks):
            if y_trues[i] <= cur_upper[i] and y_trues[i] >= cur_lower[i]:
                covs.append(1)
            else:
                covs.append(0)
        covs = np.array(covs)
        if cumsum:
            covs = covs[5:]
            avg_covs = np.cumsum(covs)
            scale_factors = np.linspace(1, len(avg_covs), len(avg_covs))
            avg_covs = avg_covs / scale_factors
        else:
            avg_covs = np.convolve(covs, np.ones(window_size)/window_size, mode='valid')
        current_cs = np.abs(avg_covs - (1-alpha))
        cs4alphas.append(current_cs)
    cs4alphas = np.array(cs4alphas)
    avg_cs = np.mean(cs4alphas, axis=0)
    np.save(file_name, avg_cs)


def plot_avg_cov_curve(preds_in_right_format, region, week_ahead, alpha, window_size, max_steps, savefig=False, figpath=''):
    y_trues, y_preds, lowers, uppers = preds_in_right_format[region][week_ahead]
    num_weeks = len(y_trues)
    if max_steps > num_weeks:
        max_steps = num_weeks
    y_preds = y_preds[:max_steps]
    y_trues = y_trues[:max_steps]
    cur_lower = lowers[alpha][:max_steps]
    cur_upper = uppers[alpha][:max_steps]
    
    covs = []
    for i in range(max_steps):
        if y_trues[i] <= cur_upper[i] and y_trues[i] >= cur_lower[i]:
            covs.append(1)
        else:
            covs.append(0)
    covs = np.array(covs)
    avg_covs = np.convolve(covs, np.ones(window_size)/window_size, mode='same')
    week_ids = np.linspace(1, len(avg_covs), len(avg_covs))
    
    plt.clf()
    plt.ylim(0, 1)
    plt.plot(week_ids, [1-alpha for i in range(len(avg_covs))], label='1 - alpha')
    plt.plot(week_ids, avg_covs, '-x', label='avg cov')
    plt.xticks(week_ids, rotation=45, ha='right')
    plt.legend()
    if savefig:
        plt.savefig(figpath)
    else:
        plt.show()


def plot_predictions_with_cis(preds_in_right_format, region, week_ahead, alpha_list, max_steps=1000, fill=True, savefig=False, figpath=''):
    y_trues, y_preds, lowers, uppers = preds_in_right_format[region][week_ahead]
    num_weeks = len(y_trues)
    week_ids = np.linspace(1, num_weeks, num_weeks)
    if max_steps > num_weeks:
        max_steps = num_weeks
    week_ids = week_ids[:max_steps]
    plt.figure(figsize=(20, 4))
    plt.plot(week_ids, y_preds[:max_steps], '-o', label='prediction')
    plt.plot(week_ids, y_trues[:max_steps], '-o', label='ground truth')
    for alpha in alpha_list:
        cur_lower = lowers[alpha][:max_steps]
        cur_upper = uppers[alpha][:max_steps]
        if fill:
            plt.fill_between(week_ids, cur_lower, cur_upper, alpha=0.3)
        else:
            plt.plot(week_ids, cur_lower, label=f'{alpha}_lower')
            plt.plot(week_ids, cur_upper, label=f'{alpha}_upper')
    plt.xticks(week_ids, rotation=45, ha='right')
    plt.legend()
    if savefig:
        plt.savefig(figpath)
    else:
        plt.show()


def plot_losses(loss_file_path, loss_types=['total_loss'], plot_train_loss=True, plot_val_loss=False):
    """
    saved_losses: dict:
        train_loss -> {total_loss -> [], qloss -> [], closs -> [], eloss -> [], mloss -> []}
        val_loss -> {total_loss -> [], qloss -> [], closs -> [], eloss -> [], mloss -> []} 
    """
    saved_losses = pickle_load(loss_file_path, version5=True)
    
    def plot_loss_helper(losses, loss_types, loss_source):
        """
        loss: {total_loss -> [], qloss -> [], closs -> [], eloss -> [], mloss -> []}
        """
        epochs = None
        for loss_type, data in losses.items():
            if epochs is None:
                epochs = np.linspace(1, len(data), len(data))
            if loss_type in loss_types:
                plt.plot(epochs, data, label=f'{loss_source}: {loss_type}', marker='o')

    # train loss
    if plot_train_loss:
        plot_loss_helper(saved_losses['train_loss'], loss_types, 'train')
    
    # val loss
    if plot_val_loss:
        plot_loss_helper(saved_losses['val_loss'], loss_types, 'val')
    
    plt.legend()
    plt.show()


def plot_ground_truth(data_file_id, region, week_ahead_idx, start_week_idx, end_week_idx):
    all_base_preds = pickle_load(f'../../results/base_pred/saved_pred_{data_file_id}.pickle', version5=True)
    base_pred = all_base_preds['base_pred']
    y_trues = []
    y_preds = []
    week_ids = []
    prev_week_idx = -1
    round_idx = 0
    for i, saved_info in enumerate(base_pred):
        predictions, addition_infos = saved_info
        prediction = predictions[region]
        addition_info = addition_infos[region]
        y, y_mask, x, x_mask, week_id = addition_info
        try:
            week_id = int(week_id)
        except:
            week_id = i
        if week_id < prev_week_idx:
            round_idx += 1
        prev_week_idx = week_id
        y_true = y[week_ahead_idx]
        y_preds.append(prediction[week_ahead_idx])
        y_trues.append(y_true)
        week_ids.append(str(week_id + round_idx * 1000))
    plt.figure(figsize=(30, 4))
    plt.plot(week_ids[start_week_idx:end_week_idx], y_preds[start_week_idx:end_week_idx], '-x', label='pred')
    plt.plot(week_ids[start_week_idx:end_week_idx], y_trues[start_week_idx:end_week_idx], '-x', label='true')
    plt.xticks(week_ids[start_week_idx:end_week_idx], rotation=45, ha='right')
    plt.legend()
    plt.show()


def plot_horizon_predictions(preds, region, alphas_list, week_idx, prev_weeks=4, savefig=False, figpath=''):
    week_aheads = [1, 2, 3, 4]
    ppreds = []
    ipreds_lower = {}
    ipreds_upper = {}
    pred_xs = [i+week_idx for i in range(4)]
    true_xs = []
    for alpha in alphas_list:
        ipreds_lower[alpha] = []
        ipreds_upper[alpha] = []
    for week_ahead in week_aheads:
        _, y_preds, lowers, uppers = preds[region][week_ahead]
        ppreds.append(y_preds[week_idx])
        for alpha in alphas_list:
            ipreds_lower[alpha].append(lowers[alpha][week_idx])
            ipreds_upper[alpha].append(uppers[alpha][week_idx])
    gts = []
    # add prev weeks
    for i in range(prev_weeks):
        cur_week_idx = week_idx - prev_weeks + i
        gts.append(preds[region][1][0][cur_week_idx])
        true_xs.append(cur_week_idx)
    for ahead in week_aheads:
        gts.append(preds[region][ahead][0][cur_week_idx])
        true_xs.append(week_idx + ahead - 1)
    # add next four weeks
    # plot
    plt.clf()
    plt.plot(pred_xs, ppreds, 'o', label='prediction')
    for alpha in alphas_list:
        plt.plot(pred_xs, ipreds_lower[alpha], '-', label=f'{alpha}_lower')
        plt.plot(pred_xs, ipreds_upper[alpha], '-', label=f'{alpha}_upper')
    plt.plot(true_xs, gts, '-x', label='gt')
    plt.legend()
    if savefig:
        plt.savefig(figpath)
    else:
        plt.show()


def box_plot(step_ahead, metric_idx, metrics):
    res_list = []
    regions = list(metrics.keys())
    for region in regions:
        res_list.append(metrics[region][step_ahead][metric_idx])
    plt.boxplot(res_list)
    plt.show()


def sort_preds1(preds, regions, week_aheads):
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


def cs_curve(step_ahead, region, preds, sorting=True, save_path=None):
    (y_trues, y_preds, lowers, uppers) = preds[region][step_ahead]
    alpha2coverage = {}
    for alpha in lowers:
        in_range_count = 0
        for i in range(len(y_preds)):
            if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
                in_range_count += 1
        cov_rate = in_range_count / len(y_preds)
        alpha2coverage[1-alpha] = cov_rate
    plt.clf()
    plt.figure(figsize=(3.6, 3.6))
    plt.plot(list(alpha2coverage.keys()), list(alpha2coverage.keys()), '--', label='ideal', color='black')
    
    if sorting:
        plt.plot(list(alpha2coverage.keys()), list(alpha2coverage.values()), '-', label='unsorted', color='blue', marker='x')
        # sorted results
        sorted_preds = sort_preds1(preds, [region], [step_ahead])
        (y_trues, y_preds, lowers, uppers) = sorted_preds[region][step_ahead]
        alpha2coverage_sorted = {}
        for alpha in lowers:
            in_range_count = 0
            for i in range(len(y_preds)):
                if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
                    in_range_count += 1
            cov_rate = in_range_count / len(y_preds)
            alpha2coverage_sorted[1-alpha] = cov_rate
        plt.plot(list(alpha2coverage.keys()), list(alpha2coverage_sorted.values()), '-', label='sorted', color='red')
    else:
        plt.plot(list(alpha2coverage.keys()), list(alpha2coverage.values()), '-', label='realized', color='red', marker='x')
    
    plt.xlabel('Desired coverage')
    plt.ylabel('Realized coverage')
    plt.legend()
    plt.axis([0, 1, 0, 1])
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()    
        

def plot_cov_pred_in_one_figure(preds_in_right_format, region, week_ahead, alpha, window_size=11, max_steps=1000, fill=True, figsize=(10, 6), save_path=None):
    y_trues, y_preds, lowers, uppers = preds_in_right_format[region][week_ahead]
    num_weeks = len(y_trues)
    steps = min(max_steps, num_weeks)
    
    cur_y_trues = np.array(y_trues[:steps])
    cur_y_preds = np.array(y_preds[:steps])
    cur_lower = np.array(lowers[alpha][:steps])
    cur_upper = np.array(uppers[alpha][:steps])
    
    # cov
    covs = np.array([(1 if (cur_lower[i] <= y_trues[i] <= cur_upper[i]) else 0) for i in range(steps)])
    avg_covs = np.convolve(covs, np.ones(window_size)/window_size, mode='valid')
    
    # predictions
    tmp_idxes = np.ones(len(cur_y_trues)) * 10
    cov_tmp_idxes = np.convolve(tmp_idxes, np.ones(window_size)/window_size, mode='same')
    valid_idxes = np.where(cov_tmp_idxes == tmp_idxes)
    valid_y_trues = cur_y_trues[valid_idxes]
    valid_y_preds = cur_y_preds[valid_idxes]
    valid_lower = cur_lower[valid_idxes]
    valid_upper = cur_upper[valid_idxes]
    
    week_ids = np.arange(1, len(valid_upper)+1, dtype=int)
    
    # plot
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # cov plot
    ax1.set_ylim(0, 1)
    ax1.plot(week_ids, [1 - alpha] * len(avg_covs), label='target coverage')
    ax1.plot(week_ids, avg_covs, '-x', label='running emperical coverage')
    ax1.set_xticklabels([])

    # Optionally, also remove the x-axis label and ticks for a cleaner look:
    ax1.set_xlabel('')
    ax1.set_xticks([])  # Removes tick marks
    ax1.legend()
    
    # predictions
    ax2.plot(week_ids, valid_y_preds[:steps], '-o', label='prediction')
    ax2.plot(week_ids, valid_y_trues[:steps], '-o', label='ground truth')
    ax2.fill_between(week_ids, valid_lower, valid_upper, alpha=0.3)
    ax2.set_xticks(week_ids)
    ax2.legend()
    
    ax2.set_ylabel('Flu hosp')
    ax2.set_xlabel('Week number')  # Add if desired

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()


######### one experiment, all methods together #########


def prediction_intervals_plot(all_preds, region, week_ahead, alpha=0.1):
    methods = list(all_preds.keys())
    fig, axes = plt.subplots(len(methods), 1)
    fig.set_figheight(10)
    fig.set_figwidth(6)
    fig.tight_layout()
    for i in range(len(methods)):
        current_method = methods[i]
        y_trues, y_preds, all_lowers, all_uppers = all_preds[current_method][region][week_ahead]
        lowers = all_lowers[alpha]
        uppers = all_uppers[alpha]
        idxes = np.linspace(1, len(y_trues), len(y_trues))
        axes[i].plot(idxes, y_trues, label='ground truth')
        axes[i].plot(idxes, y_preds, label='prediction')
        axes[i].fill_between(idxes, lowers, uppers, alpha=0.3)
        if i == 0:
            axes[i].legend()
        axes[i].set_title(f'{current_method}')
    plt.show()


def running_coverage_plot(all_preds, region, week_ahead, alpha=0.1, window_size=15):
    methods = list(all_preds.keys())
    fig, axes = plt.subplots(len(methods), 1)
    fig.set_figheight(10)
    fig.set_figwidth(6)
    fig.tight_layout()
    for i in range(len(methods)):
        current_method = methods[i]
        y_trues, y_preds, all_lowers, all_uppers = all_preds[current_method][region][week_ahead]
        lowers = all_lowers[alpha]
        uppers = all_uppers[alpha]
        
        covs = np.array([(1 if (lowers[i] <= y_trues[i] <= uppers[i]) else 0) for i in range(len(y_trues))])
        avg_covs = np.convolve(covs, np.ones(window_size)/window_size, mode='valid')
        idxes = np.linspace(1, len(avg_covs), len(avg_covs))
        
        axes[i].plot(idxes, avg_covs, label='average coverage')
        axes[i].plot(idxes, [1 - alpha]*len(avg_covs), label='target coverage', linestyle='--')
        if i == 0:
            axes[i].legend()
        axes[i].set_title(f'{current_method}')
    plt.show()