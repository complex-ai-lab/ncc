from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from forecaster.cpmethods import prepare_scores
from forecaster.utils import pickle_load, pickle_save


def clip_base_pred_using_index(base_pred, index_list):
    print('aaa')
    print(index_list)
    # print(len(base_pred))
    new_base_pred = [base_pred[i] for i in index_list]
    return new_base_pred


def prepare_index_list(se_list):
    index_list = []
    for se in se_list:
        # [start, end)
        start_index = se[0]
        end_index = se[1] 
        index_list += list(range(start_index, end_index))
    return index_list


def clip_base_pred(base_pred, skip_beginning=0, skip_last=0):
    print(skip_beginning)
    print(skip_last)
    new_base_pred = base_pred[skip_beginning:]
    new_base_pred = new_base_pred[:len(new_base_pred)-skip_last]
    return new_base_pred


# python check_base_pred.py -d=4 -c -n=5 -b=62 -l=0
# python check_base_pred.py -d=4
if __name__ == '__main__':
    target_region = 'US'
    ahead = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', '-d')
    parser.add_argument('--clip', '-c', action='store_true')
    
    parser.add_argument('--use_cif', '-u', action='store_true')
    parser.add_argument('--clip_index_file', '-f')
    
    parser.add_argument('--skip_beginning', '-b')
    parser.add_argument('--skip_last', '-l')
    parser.add_argument('--new_data_id', '-n')
    args = parser.parse_args()
    
    data_id = int(args.data_id)
    saved_pred = pickle_load(f'../../results/base_pred/saved_pred_{data_id}.pickle', version5=True)
    base_pred = saved_pred['base_pred']
    
    if args.clip:
        if args.use_cif:
            clip_index_list = pickle_load(args.clip_index_file, version5=True)
            new_base_pred = clip_base_pred_using_index(base_pred, clip_index_list)
        else:
            new_base_pred = clip_base_pred(base_pred, skip_beginning=int(args.skip_beginning), skip_last=int(args.skip_last))
        new_saved_pred = saved_pred
        new_saved_pred['base_pred'] = new_base_pred
        pickle_save(f'../../results/base_pred/saved_pred_{args.new_data_id}.pickle', new_saved_pred)
    else:
        scores, y_preds, y_trues = prepare_scores(base_pred, target_region=target_region, ahead=ahead)
        print(f'number of weeks is {len(y_preds)}')
        
        # plot ground truth v.s. predictions
        plt.plot(y_preds, '-x', label='pred')
        plt.plot(y_trues, label='true')
        plt.legend()
        plt.show()
        
        # plot scores
        plt.plot(scores)
        plt.show()
    