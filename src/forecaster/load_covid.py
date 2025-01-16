from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
from epiweeks import Week
import numpy as np

from forecaster.data_utils import get_state_train_data, create_window_seqs, create_fixed_window_seqs, prepare_ds, get_state_test_data_xy

import warnings
warnings.filterwarnings("ignore")



def prepare_data(params):
    # predict [pred_week + 1, pred_week + weeks_ahead]
    
    # get params
    smooth = params['smooth']
    fix_window = params['fix_window']
    pred_week = Week.fromstring(params['last_train_time'])
    
    seq_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state
    ys_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state

    region_idx = {r: i for i, r in enumerate(params['regions'])}

    def one_hot(idx, dim=len(region_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans.reshape(1, -1)
    
    test_y_avail = True
    if params['test_time'] == params['last_train_time']:
        test_y_avail = False

    seq_length = 0
    
    # define lists
    train_regions, train_metas, train_xs, train_xs_masks, train_ys, train_ys_mask = [], [], [], [], [], []
    test_regions, test_metas, test_xs, test_xs_masks, test_ys, test_ys_mask = [], [], [], [], [], []

    for region in params['regions']:
        x, y = get_state_train_data(params, region, smooth)
        x_tmp = seq_scalers[region].fit_transform(x.values[:, :-1])
        x = np.concatenate((x_tmp, x.values[:, -1:]), axis=-1)
        y = ys_scalers[region].fit_transform(y)

        x, x_mask, y, y_mask = create_fixed_window_seqs(
            x, y, params['data_params']['min_sequence_length'],
            params['weeks_ahead'], params['data_params']['pad_value']) if fix_window else create_window_seqs(
                x, y, params['data_params']['min_sequence_length'],
                params['weeks_ahead'], params['data_params']['pad_value'])

        train_regions.extend([region] * x.shape[0])
        train_metas.append(
            np.repeat(one_hot(region_idx[region]), x.shape[0], axis=0))
        train_xs.append(x.astype(np.float32))
        train_xs_masks.append(x_mask.astype(np.float32))
        train_ys.append(y)
        train_ys_mask.append(y_mask)
        
        seq_length = len(x_mask[0])

        if not test_y_avail:
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append(x[[-1]].astype(np.float32))
            test_xs_masks.append(x_mask[[-1]].astype(np.float32))
        else:
            x_test, y_test = get_state_test_data_xy(params, region, pred_week, seq_length, smooth)
            x_test = x_test.to_numpy()
            x_tmp = seq_scalers[region].transform(x_test[:, :-1])
            x_test = np.concatenate((x_tmp, x_test[:, -1:]), axis=-1)
            y_test = ys_scalers[region].transform(y_test)
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append([x_test.astype(np.float32)])
            test_xs_masks.append([np.zeros(seq_length).astype(np.float32)])
            test_ys.append([y_test])
            test_ys_mask.append([np.ones_like(y_test)])

    # construct dataset
    dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas)
    test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, test=not test_y_avail)

    # split train dataset into train and validation
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create DataLoader for training data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=True)

    # Create DataLoader for validation data
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=False)

    # Create DataLoader for test data
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=False)

    train_xs = np.concatenate(train_xs, axis=0)
    return train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers, seq_length


def prepare_region_fine_tuning_data(params):
    all_dataloaders = {}
    # predict [pred_week + 1, pred_week + weeks_ahead]
    
    # get params
    smooth = params['smooth']
    fix_window = params['fix_window']
    pred_week = Week.fromstring(params['last_train_time'])
    
    seq_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state
    ys_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state

    region_idx = {r: i for i, r in enumerate(params['regions'])}

    def one_hot(idx, dim=len(region_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans.reshape(1, -1)
    
    test_y_avail = True
    if params['test_time'] == params['last_train_time']:
        test_y_avail = False
    
    for region in params['regions']:
        # define lists
        train_regions, train_metas, train_xs, train_xs_masks, train_ys, train_ys_mask = [], [], [], [], [], []
        test_regions, test_metas, test_xs, test_xs_masks, test_ys, test_ys_mask = [], [], [], [], [], []
        
        x, y = get_state_train_data(params, region, smooth)
        x_tmp = seq_scalers[region].fit_transform(x.values[:, :-1])
        x = np.concatenate((x_tmp, x.values[:, -1:]), axis=-1)
        y = ys_scalers[region].fit_transform(y)

        x, x_mask, y, y_mask = create_fixed_window_seqs(
            x, y, params['data_params']['min_sequence_length'],
            params['weeks_ahead'], params['data_params']['pad_value']) if fix_window else create_window_seqs(
                x, y, params['data_params']['min_sequence_length'],
                params['weeks_ahead'], params['data_params']['pad_value'])

        train_regions.extend([region] * x.shape[0])
        train_metas.append(
            np.repeat(one_hot(region_idx[region]), x.shape[0], axis=0))
        train_xs.append(x.astype(np.float32))
        train_xs_masks.append(x_mask.astype(np.float32))
        train_ys.append(y)
        train_ys_mask.append(y_mask)
        
        seq_length = len(x_mask[0])

        if not test_y_avail:
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append(x[[-1]].astype(np.float32))
            test_xs_masks.append(x_mask[[-1]].astype(np.float32))
        else:
            x_test, y_test = get_state_test_data_xy(params, region, pred_week, seq_length, smooth)
            x_test = x_test.to_numpy()
            x_tmp = seq_scalers[region].transform(x_test[:, :-1])
            x_test = np.concatenate((x_tmp, x_test[:, -1:]), axis=-1)
            y_test = ys_scalers[region].transform(y_test)
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append([x_test.astype(np.float32)])
            test_xs_masks.append([np.zeros(seq_length, dtype=float)])
            test_ys.append([y_test])
            test_ys_mask.append([np.ones_like(y_test)])

        # construct dataset
        dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas)
        test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, test=not test_y_avail)

        # split train dataset into train and validation
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

        # Create DataLoader for training data
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['rft_batch_size'],
            shuffle=True)

        # Create DataLoader for validation data
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['rft_batch_size'],
            shuffle=False)

        # Create DataLoader for test data
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False)

        train_xs = np.concatenate(train_xs, axis=0)
        all_dataloaders[region] = (train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers)

    return all_dataloaders