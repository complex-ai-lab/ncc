import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from forecaster.utils import pickle_load
from forecaster.e2ecp import load_all_params
from forecaster.eval_cp import eval_one_exp


FONT_SIZE = 16
fig_width = 5.6
fig_height = 5


OURMETHOD = 'NCC'
baseline_names = ['nexcp', 'cfrnn', 'aci', 'pid', OURMETHOD]
display_names = ['NEXCP', 'CF-RNN', 'ACI', 'C-PID', OURMETHOD]

dataset2color = {
    'covid': '^',
    'flu': 'o',
    'smd': 's',
    'elec': 'd',
    'weather': 'v',
}

model2mark = {
    'nexcp': 'blue',
    'faci': 'black',
    'cfrnn': 'green',
    'aci': 'orange',
    'pid': 'purple',
    OURMETHOD: 'red',
}


def load_results_from_csv_helper(expid, model_name, metrics:list):
    current_name = model_name if model_name != OURMETHOD else 'multi_seeds'
    fname = f'../../results/csvs/{expid}_{current_name}.csv'
    df = pd.read_csv(fname, header=0)
    vals = []
    for metric in metrics:
        if model_name == OURMETHOD:
            metric_val = df.loc[(df['seed'] == 'mean') & (df['week ahead'] == 1), metric].to_numpy().item()
        else:
            metric_val = df.loc[(df['region'] == 'mean') & (df['week ahead'] == 1), metric].to_numpy().item()
        vals.append(metric_val)
    return vals


# header = ['seed', 'week ahead', 'rmse', 'nrmse', 'mape', 'cs_score', 'norm_crps_val', 'mean_wis', 'mono', 'cov_50', 'cov_95', 'cov_90', 'pi_length_90', 'dvs']
        
# WriteRows2CSV(rows, f'../../results/{input_id}_multi_seeds.csv')

def plot_two_metrics_load_results_from_csv(dataset2expid:dict, metric_name_x:str = 'cs_score', metric_name_y:str = 'dvs'):
    plt.figure(figsize=(fig_width, fig_height))
    
    scatter_plots = []
    for dataset in dataset2expid:
        expid = dataset2expid[dataset]
        current_scatter_plots = []
        current_data = []
        for model_name in baseline_names:
            xy_vals = load_results_from_csv_helper(expid, model_name, [metric_name_x, metric_name_y])
            current_data.append(xy_vals)
        best_x_metric = 1 # smaller the better
        for xy_vals in current_data:
            if xy_vals[0] < best_x_metric:
                best_x_metric = xy_vals[0]
        for i, model_name in enumerate(baseline_names):
            xy_vals = current_data[i]
            xy_vals[0] = xy_vals[0] - best_x_metric
            scatter_plot = plt.scatter(xy_vals[0], xy_vals[1], marker=dataset2color[dataset], c=model2mark[model_name], s=66)
            current_scatter_plots.append(scatter_plot)
        scatter_plots.append(current_scatter_plots)
        
     # Add a shaded area (rectangle). Adjust (x, y) and (width, height) as needed.
    ax = plt.gca()
    x_min, _ = ax.get_xlim()
    _, y_max = ax.get_ylim()
    cs_max = 0.01
    dcs_min = 0.8
    triangle_vertices = [(x_min, y_max), (x_min, dcs_min), (cs_max, y_max)]
    triangle = patches.Polygon(triangle_vertices, closed=True, linewidth=0, facecolor='pink', alpha=0.3)

    # Add the triangle to the plot
    ax.add_patch(triangle)
    
    # Add annotation text pointing to the shaded area with an arrow
    ax.annotate(
        'Best performance area', 
        xy=((cs_max+x_min)/2, (dcs_min+y_max)/2),  # Pointing location inside the shaded area
        xytext=(0.015, 0.66),  # Text location
        fontsize=FONT_SIZE, 
        color='black',
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4)
    )
        
        
    legend1 = plt.legend(scatter_plots[0], display_names, loc=4, fontsize=12.4)
    plt.xlabel('Calibration Score', fontsize=FONT_SIZE)
    plt.ylabel('Distribution Consistency Score', fontsize=FONT_SIZE)
    plt.legend([l[0] for l in scatter_plots], list(dataset2expid.keys()), loc=8, fontsize=12.4)
    plt.gca().add_artist(legend1)
    plt.savefig('../../results/tmp/z_2d_plot.pdf', format='pdf', bbox_inches='tight')


def run_evaluation(dataset2expid, sort=True, only_valid_pil=False):
    methods = ['nexcp', 'cfrnn', 'aci', 'pid', OURMETHOD]
    sort_option = ' '
    if sort:
        sort_option = ' -o'
    ovp_option = ' '
    if only_valid_pil:
        ovp_option = ' -ov'
    for dataset, expid in dataset2expid.items():
        clip_option = ' -c'
        if dataset == 'elec':
            clip_option = ' '
        print(f'Start evaluating dataset {dataset}')
        for method in methods:
            if method == OURMETHOD:
                os.system(f'python eval_cp.py -m={method} -i={expid} -u -p=flu {clip_option} {sort_option} {ovp_option}')
            else:
                os.system(f'python eval_cp.py -m={method} -i={expid} -s=1 -p={dataset} {clip_option} {sort_option} {ovp_option}')


def main():
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
    
    newflu1 = {
        'flu': 28202,
    }
    newflu2 = {
        'flu': 28203,
    }
    newflu3 = {
        'flu': 28204,
    }
    newflu4 = {
        'flu': 28205,
    }
    
    # run_evaluation(newflu1)
    # run_evaluation(newflu2)
    # run_evaluation(newflu3)
    # run_evaluation(newflu4)
    
    # run_evaluation(dataset2expid_informer, sort=True, only_valid_pil=False)
    # run_evaluation(dataset2expid_theta, sort=True, only_valid_pil=False)
    
    # run_evaluation(dataset2expid_seq2seq, sort=False, only_valid_pil=False)
    # run_evaluation(dataset2expid_2ahead, sort=True, only_valid_pil=False)
    # run_evaluation(dataset2expid_3ahead, sort=True, only_valid_pil=False)
    # run_evaluation(dataset2expid_4ahead, sort=True, only_valid_pil=False)
    
    # run_evaluation(dataset2expid_seq2seq, sort=False, only_valid_pil=False)
    plot_two_metrics_load_results_from_csv(dataset2expid_seq2seq)
    # run_evaluation(dataset2expid_seq2seq, sort=True, only_valid_pil=False)


if __name__ == '__main__':
    main()

