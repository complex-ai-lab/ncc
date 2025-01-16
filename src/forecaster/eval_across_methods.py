import matplotlib.pyplot as plt
import numpy as np
import os


from forecaster.utils import pickle_load
from forecaster.eval_across_exps import FONT_SIZE, fig_width, fig_height


methods = ['nexcp', 'cfrnn', 'aci', 'pid', 'e2ecp']
display_names = [
    'NEXCP',
    'CF-RNN',
    'ACI',
    'C-PID',
    'NCC'
]# names displayed on the plot
model2color = {
    'nexcp': 'blue',
    'faci': 'black',
    'cfrnn': 'green',
    'aci': 'orange',
    'pid': 'purple',
    'e2ecp': 'red',
}

def running_cs_plot(exp_id, suffix='avg_cs'):
    folder_path = f'../../results/'
    data_file_list = [folder_path + f'{exp_id}_{method}_{suffix}.npy' for method in methods] # a list of saved pickle files
    cmap = plt.get_cmap('hsv')  # You can choose any colormap like 'viridis', 'plasma', etc.
    for i in range(len(data_file_list)):
        current_data = np.load(data_file_list[i])
        current_name = display_names[i]
        if current_name == 'NCC':
            plt.plot(current_data, label=current_name, color='red')
        else:
            plt.plot(current_data, label=current_name, alpha=0.5, color=model2color[methods[i]])
    plt.legend()
    plt.xlabel('Week number', fontsize=FONT_SIZE)
    plt.ylabel('Calibration score', fontsize=FONT_SIZE)
    plt.savefig('../../results/tmp/transfer_learning.pdf', format='pdf', bbox_inches='tight')


def pil_plot(exp_id, suffix='90pil'):
    folder_path = f'../../results/'
    data_file_list = [
        folder_path + f'{exp_id}_{method}_{suffix}.pkl' for method in methods
    ] # a list of saved pickle files
    fig, axes = plt.subplots(len(methods), 1)
    fig.set_figheight(10)
    fig.set_figwidth(6)
    fig.tight_layout()
    for i in range(len(data_file_list)):
        current_data = pickle_load(data_file_list[i])
        current_name = display_names[i]
        y_trues = current_data['y_true']
        y_preds = current_data['y_pred']
        lowers = current_data['lower']
        uppers = current_data['upper']
        idxes = np.linspace(1, len(y_trues), len(y_trues))
        axes[i].plot(idxes, y_trues, label='ground truth')
        axes[i].plot(idxes, y_preds, label='prediction')
        axes[i].fill_between(idxes, lowers, uppers, alpha=0.3)
        if i == 0:
            axes[i].legend(fontsize=FONT_SIZE-6)
        axes[i].set_title(f'{current_name}', fontsize=FONT_SIZE-6)
    # plt.xlabel('data index')
    plt.savefig('../../results/tmp/zsmd_pil.pdf', format='pdf', bbox_inches='tight')



mono_data = {
    'datasets': ['flu', 'covid-19', 'weather', 'smd', 'elec'],
    'NCC': [0.90, 1, 1, 1, 1],
    'NCC-T': [0.88, 1, 1, 1, 1],
    'NCC-M': [0.80, 1, 1, 1, 1],
}

def mono_bar_plot():
    x_size = 1
    entities = mono_data['datasets'][:x_size]
    values1 = mono_data['NCC'][:x_size]  # Values for the first bar
    values2 = mono_data['NCC-T'][:x_size]  # Values for the second bar
    values3 = mono_data['NCC-M'][:x_size]  # Values for the third bar

    # Create an array with positions of the entities
    x = np.arange(len(entities))
    

    # Width of the bars
    width = 0.04

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(3, fig_height))
    bar1 = ax.bar(x - 1.2*width, values1, width=width, label='NCC')
    bar2 = ax.bar(x,         values2, width=width, label='NCC-T')
    bar3 = ax.bar(x + 1.2*width, values3, width=width, label='NCC-M')
    # ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(bottom=0.7)
    # Add labels, title, and custom ticks
    ax.set_xlabel('Datasets', fontsize=FONT_SIZE)
    ax.set_ylabel('DCS', fontsize=FONT_SIZE)
    # ax.set_title('Distribution consistency score for DCC and DCC-M (DCC without monotonicity loss)')
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.legend(loc='upper right')

    # Display the plot
    plt.savefig('../../results/tmp/zmono_bar1.pdf', format='pdf', bbox_inches='tight')


def mono_bar_plot_on_flu():
    entities = ['NCC', 'NCC-T', 'NCC-M']
    values = [0.95, 0.90, 0.80]

    # Adjust the positions of the entities to remove spaces between bars
    x = np.arange(len(entities))

    # Set the width of the bars to 1 so they fill the space
    width = 1.0

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(1, 5))
    bar = ax.bar(x, values, width)

    # Set y-axis limits
    ax.set_ylim(bottom=0.7)

    # Add labels and custom ticks
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('DCS', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=12)

    # Save the plot
    plt.savefig('../../results/tmp/zmono_bar_flu.pdf', format='pdf', bbox_inches='tight')


data = {
    'Base forecaster': ['Seq2seq', 'Informer', 'ThetaModel'],
    'values1': [5, 4, 4],
    'methods1': ['NCC'] * 3,
    'values2': [0, 1, 1],
    'methods2': ['', 'ACI', 'ACI']
}

def more_base_model_bar_plot(data):
    # Extracting data
    base_forecaster = data['Base forecaster']
    values1 = data['values1']
    methods1 = data['methods1']
    values2 = data['values2']
    methods2 = data['methods2']

    # Setting the positions and width for the bars
    x = np.arange(len(base_forecaster))
    width = 0.35

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar1 = ax.bar(x - width/2, values1, width, label='Methods1', color='red')
    bar2 = ax.bar(x + width/2, values2, width, label='Methods2', color='orange')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Base Forecaster', fontsize=FONT_SIZE)
    ax.set_ylabel('Number of datasets', fontsize=FONT_SIZE)  # Adding ylabel
    # ax.set_title('Values and Methods for Base Forecaster')
    ax.set_xticks(x)
    ax.set_xticklabels(base_forecaster, fontsize=FONT_SIZE)

    # Labeling the bars with the corresponding method
    for rect, label in zip(bar1, methods1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, label,
                ha='center', va='bottom', fontsize=FONT_SIZE-3.2)

    for rect, label in zip(bar2, methods2):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, label,
                ha='center', va='bottom', fontsize=FONT_SIZE-3.2)

    # Show legend
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.tight_layout()

    # Save the plot (optional)
    plt.savefig('../../results/tmp/z_more_base_f.pdf', format='pdf', bbox_inches='tight')

    # Show the plot
    # plt.show()



if __name__ == "__main__":
    # run commands
    exp_id = 23005
    # for method in methods:
    #     os.system(f'python eval_cp.py -m={method} -s=2 -p=covid -i={exp_id} -o')
    plt.figure(figsize=(fig_width, fig_height))
    running_cs_plot(exp_id=exp_id, suffix='avg_cs')
    
    # pil plot, exp = 22053
    exp_id = 22051
    # for method in methods:
    #     os.system(f'python eval_cp.py -m={method} -i={exp_id} -s=1 -p=smd -c -o')
    # pil_plot(exp_id)
    
    # bar plot for monotonicity ablation study
    # mono_bar_plot(0, None)
    # mono_bar_plot_on_flu()
    more_base_model_bar_plot(data=data)
    
    mono_bar_plot()