# read in <machine-1-4.txt> and convert it to csv with selected columns
import numpy as np
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def WriteRows2CSV(rows:list, output_path:Path):
    with open(output_path, 'w+') as f: 
        csv_writer = csv.writer(f)  
        csv_writer.writerows(rows)


if __name__ == '__main__':
    
    #########
    # setup #
    #########

    output_file = Path('.csv')
    target_columns = [1]
    header = ['simdata']
    
    # read txt file
    smd_data = np.genfromtxt(input_file, dtype=np.float32, delimiter=',', skip_header=skip_header)
    
    # only use column 2, 3, 4
    smd_data = smd_data[:, target_columns]
    
    # validate
    print(smd_data.shape)
    smd_data4plot = smd_data[:100]
    for i in range(smd_data4plot.shape[1]):
        plt.plot(smd_data4plot[:, i], label=f'{i}')
    plt.legend()
    plt.show()
    
    # target is last column, write to csv
    rows = []
    rows.append(header)
    for i in range(smd_data.shape[0]):
        current_row = []
        for j in range(smd_data.shape[1]):
            current_row.append(smd_data[i, j])
        rows.append(current_row)
    WriteRows2CSV(rows, output_file)