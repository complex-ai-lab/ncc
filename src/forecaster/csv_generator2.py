# This file generates for the results table based on csvs in results/csvs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecaster.eval_cp import WriteRows2CSV

FONT_SIZE = 16

OURMETHOD = 'NCC'
# method_names = ['nexcp', 'faci', 'cfrnn', 'aci', 'pid', OURMETHOD]
method_names = ['nexcp', 'cfrnn', 'aci', 'pid', OURMETHOD]
display_names = ['NEXCP', 'CF-RNN', 'ACI', 'C-PID', OURMETHOD]
datasets = ['covid', 'flu', 'smd', 'elec', 'weather']
datasets_display = ['covid-19', 'flu', 'smd', 'electric', 'weather']
aheads = [i+1 for i in range(4)]

elec_ahead_lookup = {
    1: 1,
    2: 3,
    3: 6,
    4: 12,
}

weather_ahead_lookup = {
    1: 1,
    2: 3,
    3: 5,
    4: 10,
}

def aheads_2_real_aheads(ahead, dataset):
    if dataset == 'elec':
        return elec_ahead_lookup[ahead]
    if dataset == 'weather':
        return weather_ahead_lookup[ahead]
    return ahead

model2color = {
    'nexcp': 'blue',
    'faci': 'black',
    'cfrnn': 'green',
    'aci': 'orange',
    'pid': 'purple',
    OURMETHOD: 'red',
}

model2mark = {    
    'nexcp': '^',
    'cfrnn': 'v',
    'aci': 's',
    'pid': 'o',
    OURMETHOD: 'x',
}

###########################
# DEFINE DICTIONARYS HERE #
###########################

dataset2expid_seq2seq = {
    'covid': 22055,
    # 'flu': 22054,
    'smd': 22053,
    'elec': 22052,
    'weather': 22051,
    'flu': 28200,
}

dataset2expid_2ahead = {
    'covid': 29005,
    # 'flu': 29004,
    'smd': 29003,
    'elec': 29020,
    'weather': 29012,
    'flu': 28200,
}

dataset2expid_3ahead = {
    'covid': 29105,
    # 'flu': 29104,
    'smd': 29130,
    'elec': 29021,
    'weather': 29101,
    'flu': 28200,
}

dataset2expid_4ahead = {
    'covid': 29205,
    # 'flu': 29204,
    'smd': 29203,
    'elec': 29202, # ours is not 29202, fix manually
    'weather': 29014,
    'flu': 28200
}


lookup_table = {
    1: dataset2expid_seq2seq,
    2: dataset2expid_2ahead,
    3: dataset2expid_3ahead,
    4: dataset2expid_4ahead,
}

missing_exps = []


###########################
# Informer and ThetaModel #
###########################

dataset2expid_informer = {
        'covid': 22329,
        'flu': 22304,
        'smd': 22303,
        'elec': 22302,
        'weather': 22301,
}
dataset2expid_theta = {
    'covid': 26005,
    'flu': 26004,
    'smd': 26003,
    'elec': 26002,
    'weather': 26001,
}


def error_bar_helper(vals:list):
    std_dev = np.std(vals, ddof=1)
    return std_dev
    

def load_results_from_csv(expid, model_name, metrics:list, ahead, seed_num=4):
    current_name = model_name if model_name != OURMETHOD else 'multi_seeds'
    fname = f'../../results/csvs/{expid}_{current_name}.csv'
    try:
        df = pd.read_csv(fname, header=0)
    except:
        print(f'{fname} does not exist.')
        if expid not in missing_exps:
            missing_exps.append(expid)
        return [999]*len(metrics), [999]*len(metrics)
    vals = []
    errbar_vals = []
    for metric in metrics:
        if model_name == OURMETHOD:
            # print(f'{expid}, {model_name}, {ahead}')
            # print(df.loc[(df['seed'] == 'mean') & (df['week ahead'] == ahead), metric])
            metric_val = df.loc[(df['seed'] == 'mean') & (df['week ahead'] == ahead), metric].to_numpy().item()
            vals_across_seeds = []
            for seed_ in range(seed_num):
                val_cur_seed = df.loc[(df['seed'] == f'{seed_+1}') & (df['week ahead'] == ahead), metric].to_numpy().item()
                vals_across_seeds.append(val_cur_seed)
            errbar_vals.append(error_bar_helper(vals_across_seeds))
        else:
            metric_val = df.loc[(df['region'] == 'mean') & (df['week ahead'] == ahead), metric].to_numpy().item()
        vals.append(metric_val)
    if len(errbar_vals) == 0:
        errbar_vals = [999]
    return vals, errbar_vals


def csv_per_method(method_name, metric, datasets, aheads, output_file):
    rows = []
    rows.append(['dataset', 'ahead', f'{method_name}', 'error bar'])
    for dataset in datasets:
        for ahead in aheads:
            cur_expid = lookup_table[ahead][dataset]
            vals, errbar_vals = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset))
            cur_row = [dataset, ahead, vals[0], errbar_vals[0]]
            rows.append(cur_row)
    WriteRows2CSV(rows, output_file)


def csv_per_method_other_base_model(method_name, metric, datasets, basemodel, output_file):
    rows = []
    rows.append(['dataset', f'{method_name}', 'error bar'])
    ahead = 1
    for dataset in datasets:
        table = dataset2expid_informer if basemodel == 'informer' else dataset2expid_theta
        cur_expid = table[dataset]
        vals, errbar_vals = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset), seed_num=3)
        cur_row = [dataset, ahead, vals[0], errbar_vals[0]]
        rows.append(cur_row)
    WriteRows2CSV(rows, output_file)


def plot_metric_curve(metric, ahead, average=False, output_file=None):
    x_ticks = datasets
    plt.xticks(ticks=[i for i in range(len(datasets))], labels=x_ticks)
    for method_name in method_names:
        ds_vals = []
        for dataset in datasets:
            if average:
                ahead_vals = []
                for ahead in aheads:
                    cur_expid = lookup_table[ahead][dataset]
                    vals, errbar_vals = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset))
                    ahead_vals.append(vals[0])
                ds_vals.append(np.mean(ahead_vals))
            else:
                cur_expid = lookup_table[ahead][dataset]
                vals, errbar_vals = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset))
                ds_vals.append(vals[0])
        plt.plot(ds_vals, label=method_name, c=model2color[method_name], marker='x')
    plt.legend()
    plt.show()


def plot_metric_curve_per_dataset(metric, dataset, output_file=None):
    aheads = [i+1 for i in range(4)]
    real_aheads = [aheads_2_real_aheads(i+1, dataset) for i in range(4)]
    x_ticks = real_aheads
    plt.xticks(ticks=[i for i in range(len(aheads))], labels=x_ticks)
    for method_name in method_names:
        ds_vals = []
        for ahead in aheads:
            cur_expid = lookup_table[ahead][dataset]
            vals, errbar_vals = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset))
            ds_vals.append(vals[0])
        plt.plot(ds_vals, label=method_name, c=model2color[method_name], marker='x')
    plt.legend()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.clf()


def plot_metric_curve_all_datasets(metric, output_file=None, is_first=False, xlabel='Calibration Score'):
    num_datasets = len(datasets)
    fig, axes = plt.subplots(1, num_datasets, figsize=(3*num_datasets, 3))
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        aheads = [j + 1 for j in range(4)]
        real_aheads = [aheads_2_real_aheads(j + 1, dataset) for j in range(4)]
        x_ticks = real_aheads
        ax.set_xticks([k for k in range(len(aheads))])
        ax.set_xticklabels(x_ticks)
        
        for j, method_name in enumerate(method_names):
            ds_vals = []
            errorbar_vals = []
            for ahead in aheads:
                cur_expid = lookup_table[ahead][dataset]
                vals, errbar_val = load_results_from_csv(cur_expid, method_name, [metric], aheads_2_real_aheads(ahead, dataset))
                ds_vals.append(vals[0])
                errorbar_vals.append(errbar_val[0])
            transparancy_opt = 1 if method_name == OURMETHOD else 0.6
            ax.plot(ds_vals, label=display_names[j], c=model2color[method_name], marker=model2mark[method_name], alpha=transparancy_opt)
            # add error bar
            if method_name == OURMETHOD:
                lbs = [ds_vals[i] - errorbar_vals[i] for i in range(len(ds_vals))]
                ubs = [ds_vals[i] + errorbar_vals[i] for i in range(len(ds_vals))]
                ax.fill_between([i for i in range(len(lbs))], lbs, ubs, alpha=0.5, color="pink")
        if is_first:
            ax.set_title(f"{datasets_display[i]}", fontdict={'family': 'monospace'}, fontsize=FONT_SIZE)
        else:
            ax.set_xlabel('Step ahead', fontsize=FONT_SIZE)
        if i == 0:
            ax.set_ylabel(f'{xlabel}', fontsize=FONT_SIZE)
        if i == len(datasets) - 1 and not is_first:
            ax.legend(fontsize=FONT_SIZE-1)

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.clf()


def main():
    # csvs
    # metric = 'cs_score'
    # for method in method_names:
    #     csv_per_method(method, metric, datasets, aheads, f'../../results/tmp/{metric}_{method}.csv')
        
    # metric = 'mean_wis'
    # for method in method_names:
    #     csv_per_method(method, metric, datasets, aheads, f'../../results/tmp/{metric}_{method}.csv')
        
    # metric = 'norm_crps_val'
    # for method in method_names:
    #     csv_per_method(method, metric, datasets, aheads, f'../../results/tmp/{metric}_{method}.csv')

    # print('The following experiments are missing.')
    # print(missing_exps)
    
    # # plots
    # plot_metric_curve(metric, ahead=2, average=True)
    
    # # plots per dataset
    # for dataset in datasets:
    #     plot_metric_curve_per_dataset(metric=metric, dataset=dataset, output_file=f'../../results/tmp/curve_plots_no_DtACI_{metric}_{dataset}.jpeg')
    
    # plot all datasets
    metric = 'cs_score'
    plot_metric_curve_all_datasets(metric=metric, output_file=f'../../results/tmp/all_ds_curve_plots_no_DtACI_{metric}.pdf', is_first=True)
    
    metric = 'mean_wis'
    plot_metric_curve_all_datasets(metric=metric, output_file=f'../../results/tmp/all_ds_curve_plots_no_DtACI_{metric}.pdf', is_first=False, xlabel='Weighted Interval Score')
    
    # # informer
    # for method in method_names:
    #     csv_per_method_other_base_model(method, 'mean_wis', datasets, 'informer', f'../../results/tmp/informer_{metric}_{method}.csv')

    # # thetamodel
    # for method in method_names:
    #     csv_per_method_other_base_model(method, 'mean_wis', datasets, 'theta', f'../../results/tmp/thetamodel_{metric}_{method}.csv')

if __name__ == '__main__':
    main()