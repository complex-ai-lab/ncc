import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch

from forecaster.data_utils import create_window_seqs, create_fixed_window_seqs, prepare_ds


def convert_to_datetime(time_str):
    # 2022-01-26 12:30:00
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def datetime2str(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def load_all_power_data():
    """
    Before load power data, run ./bin/download_power.sh to download power data.
    """
    column_types = {
        'Date': str,
        'Time': str,
        'Global_active_power': float,
        'Global_reactive_power': float,
        'Voltage': float,
        'Global_intensity': float,
        'Sub_metering_1': float,
        'Sub_metering_2': float,
        'Sub_metering_3': float,
    }
    def get_month(ss: str):
        i = ss.find("/")
        return int(ss[i+1:ss[i+1:].find("/")+i + 1]) - 1
    def get_time_of_day(ss: str):
        hour = int(ss[:2])
        if hour < 6:
            return 0
        elif hour < 12:
            return 1
        elif hour < 18:
            return 2
        else:
            return 3
    def convert2float(x):
        try:
            x = float(x)
        except:
            x = 0.0
        return x
    df = pd.read_csv("../../data/household_power_consumption/household_power_consumption.txt", sep=';')
    df['month'] = df['Date'].apply(get_month)
    df['tod'] = df['Time'].apply(get_time_of_day)
    # TODO: remove
    # df = df.head(20000)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    for column in column_types:
        if column is not 'Date' and column is not 'Time':
            df[column] = df[column].apply(convert2float)
    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)
    return df


def load_power_df(df_original, params, start_time, end_time):
    df = df_original.copy(deep=True)
    df = df[(df["DateTime"] <= end_time) & (df["DateTime"] >= start_time)]
    return df[params['data_features']]


def get_power_train_data(df_original, params):
    """ get processed dataframe of data + target as array """
    df = df_original.copy(deep=True)
    # get start and end time
    # max_training_days = int(params['data_params']['max_training_days'])
    start_time = convert_to_datetime(params['data_params']['start_time'])
    last_train_time = convert_to_datetime(params['last_train_time'])
    # earliest_start_time = last_train_time - timedelta(days=max_training_days)
    # if start_time < earliest_start_time:
    #     start_time = earliest_start_time
    # load data
    df = load_power_df(df_original, params, start_time, last_train_time)
    # select target
    target = df.loc[:, [params['target']]].values
    return df[params['data_features']], target


def get_power_test_data_as_train_data(df_original, params):
    """ get processed dataframe of data + target as array """
    df = df_original.copy(deep=True)
    start_time = convert_to_datetime(params['last_train_time'])
    last_train_time = convert_to_datetime(params['test_time'])
    # load data
    df = load_power_df(df_original, params, start_time, last_train_time)
    # select target
    target = df.loc[:, [params['target']]].values
    return df[params['data_features']], target


def get_power_test_data(df_original, params, pred_time, seq_length):
    weeks_ahead = params['weeks_ahead']
    start_time = convert_to_datetime(params['data_params']['start_time'])

    # y
    df = load_power_df(df_original, params, start_time, pred_time + timedelta(minutes=weeks_ahead)).tail(weeks_ahead)
    target = df.loc[:, [params['target']]].values
    
    # x
    df = load_power_df(df_original, params, start_time, pred_time)
    df = df.tail(seq_length)

    return df[params['data_features']], target


def prepare_power_data(df, params):
    # predict [pred_week + 1, pred_week + weeks_ahead]

    # get params
    fix_window = params['fix_window']
    pred_time = convert_to_datetime(params['last_train_time'])
    
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
        x, y = get_power_train_data(df, params)
        x = seq_scalers[region].fit_transform(x.values[:, :])
        # x = np.concatenate((x_tmp, x.values[:, -1:]), axis=-1)
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
            x_test, y_test = get_power_test_data(df, params, pred_time, seq_length)
            x_test = x_test.to_numpy()
            x_test = seq_scalers[region].transform(x_test[:, :])
            y_test = ys_scalers[region].transform(y_test)
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append([x_test.astype(np.float32)])
            test_xs_masks.append([np.zeros(seq_length).astype(np.float32)])
            test_ys.append([y_test])
            test_ys_mask.append([np.ones_like(y_test)])
    

    # construct dataset
    dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas, with_week_id=False)
    test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, test=not test_y_avail, with_week_id=False)

    # split train dataset into train and validation
    dataset_size = len(dataset)
    if dataset_size > params['data_params']['max_training_samples']:
        dataset_size = params['data_params']['max_training_samples']
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.15)
    unused_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, unused_size])

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
    print(len(train_dataloader))
    return train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers, seq_length


def prepare_testing_power_data(df, params):
    # predict [pred_week + 1, pred_week + weeks_ahead]

    # get params
    fix_window = params['fix_window']
    pred_time = convert_to_datetime(params['last_train_time'])
    
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

    seq_length = 0
    
    # define lists
    train_regions, train_metas, train_xs, train_xs_masks, train_ys, train_ys_mask = [], [], [], [], [], []
    test_regions, test_metas, test_xs, test_xs_masks, test_ys, test_ys_mask = [], [], [], [], [], []
    
    # prepare train data
    for region in params['regions']:
        x, y = get_power_train_data(df, params)
        x = seq_scalers[region].fit_transform(x.values[:, :])
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
        
    # prepare test data as train data
    for region in params['regions']:
        x, y = get_power_test_data_as_train_data(df, params)
        x = seq_scalers[region].fit_transform(x.values[:, :])
        y = ys_scalers[region].fit_transform(y)
        x, x_mask, y, y_mask = create_fixed_window_seqs(
            x, y, params['data_params']['min_sequence_length'],
            params['weeks_ahead'], params['data_params']['pad_value']) if fix_window else create_window_seqs(
                x, y, params['data_params']['min_sequence_length'],
                params['weeks_ahead'], params['data_params']['pad_value'])
        test_regions.extend([region] * x.shape[0])
        test_metas.append(
            np.repeat(one_hot(region_idx[region]), x.shape[0], axis=0))
        test_xs.append(x.astype(np.float32))
        test_xs_masks.append(x_mask.astype(np.float32))
        test_ys.append(y)
        test_ys_mask.append(y_mask)
    

    # construct dataset
    dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas, with_week_id=False)
    test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, with_week_id=False)

    # split train dataset into train and validation
    dataset_size = len(dataset)
    if dataset_size > params['data_params']['max_training_samples']:
        dataset_size = params['data_params']['max_training_samples']
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.15)
    unused_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size, unused_size])
    
    # limit the size of test dataset to 1000 as most
    test_dataset_size = len(test_dataset)
    if test_dataset_size > 1000:
        test_size = 1000
        unused_size = test_dataset_size - 1000
        test_dataset_split, _ = torch.utils.data.random_split(test_dataset, [test_size, unused_size])

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
        test_dataset_split,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=True)

    train_xs = np.concatenate(train_xs, axis=0)
    print(len(train_dataloader))
    return train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers, seq_length    