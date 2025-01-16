import numpy as np
import torch
from epiweeks import Week
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import shift


def convert_to_epiweek(x):
    """ convert string of the format YYYYww to epiweek object """
    return Week.fromstring(str(x))


def epiweek_sub(week1, week2, max_attempts=500):
    # assumes week1 is no less than week2, week1 - week2
    for i in range(max_attempts):
        if week2 + i == week1:
            return i
    return -1


def shift_np_array(data, offset=10):
    """Shift an array along second axis and pad with zeros."""
    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        new_data[:, i] = shift(data[:, i], offset, cval=0)
    return new_data


def load_ground_truth_before_test(data_file, test_week, region, length_before_test=10):
    test_week = convert_to_epiweek(test_week)
    df = pd.read_csv(data_file, low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)

    df = df[(df["epiweek"] <= test_week) & (df["epiweek"] >= test_week - length_before_test + 1)]
    df = df.ffill()
    df = df.bfill()
    # df = df.fillna(method="ffill")
    # df = df.fillna(method="backfill")
    df = df.fillna(0)
    
    # df = df.tail(length_before_test)

    target = df.loc[:, ['flu_hospitalizations']].values

    return target



def load_df(params, region, start_week_data, pred_week, smooth=False):
    """ load data and subset to desired region and epiweeks"""

    # load and clean data
    data_file = os.path.join(params['input_files']['parent_dir'],
                             params['input_files']['weekly_data'])
    df = pd.read_csv(data_file, low_memory=False)

    # subset data using init parameters
    #columns_to_shift = df.columns[6:19]
    #df[columns_to_shift] = df.groupby('region')[columns_to_shift].shift(periods=-1)
    #df = df.groupby('region').apply(lambda group: group.iloc[:-1] if len(group) >= 1 else group).reset_index(drop=True)
    df = df.ffill()
    df = df.bfill()
    # df = df.fillna(method="ffill")
    # df = df.fillna(method="backfill")
    df = df.fillna(0)
    
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    df = df[(df["epiweek"] <= pred_week) & (df["epiweek"] >= start_week_data)]

    # smooth all features
    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode='full')[:-w + 1]

    # if target variable not in features, add it
    if params['target'] not in params['data_features']:
        params['data_features'].append(params['target'])

    # smooth all features
    if smooth:
        # add not smoothed true
        df['test_gt'] = df[params['target']].copy()
        for feature in params['data_features']:
            df.loc[:, feature] = moving_average(df.loc[:, feature].values, params['data_params']['smooth_window'])
        return df[params['data_features']+['test_gt']]
    
    df["weekid"] = df["epiweek"].apply(lambda x: int(x.cdcformat()[4:]))
    # df["weekid"] = df["epiweek"].apply(lambda x: x.cdcformat())
    
    # return subset of data
    return df[params['data_features']+['weekid']]


def get_state_train_data(params, region, smooth=False):
    """ get processed dataframe of data + target as array """

    # convert to epiweeks
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    last_train_week = convert_to_epiweek(params['last_train_time'])

    # load data
    df = load_df(params, region, start_week, last_train_week, smooth)

    # select target
    target = df.loc[:, [params['target']]].values

    return df[params['data_features']+['weekid']], target


def get_state_test_data(params, region, pred_week, smooth=False):
    """ get processed dataframe of data + target as array"""
    start_week = convert_to_epiweek(params['data_params']['start_time'])
    pred_week = convert_to_epiweek(pred_week)
    weeks_ahead = params['weeks_ahead']

    # import smoothed dataframe
    df = load_df(params, region, start_week, pred_week + weeks_ahead, smooth).tail(weeks_ahead)

    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values

    return df[params['data_features']+['weekid']], target


def get_state_test_data_xy(params, region, pred_week, x_length, smooth=False, remove_weeks_after_test=True):
    weeks_ahead = params['weeks_ahead']
    start_week = convert_to_epiweek(params['data_params']['start_time'])

    # y
    test_week = convert_to_epiweek(params['test_time'])
    avail_weeks = epiweek_sub(test_week, pred_week)
    clipped_weeks_ahead = min(avail_weeks, weeks_ahead) if remove_weeks_after_test else weeks_ahead
    df = load_df(params, region, start_week, pred_week + clipped_weeks_ahead, smooth).tail(clipped_weeks_ahead)
    target = df.loc[:, [params['target']]].values if not smooth else df.loc[:, ['test_gt']].values
    
    # pad if clipped weeks ahead is smaller than weeks ahead
    tmp_target = [-9] * weeks_ahead
    for i in range(target.shape[0]):
        tmp_target[i] = target[i, 0]
    target = np.array(tmp_target).reshape(-1, 1)
    
    # x
    df = load_df(params, region, start_week, pred_week, smooth)
    df = df.tail(x_length)

    return df[params['data_features']+['weekid']], target


def pad_sequence(seqs, batch_first=True, padding_value=0, max_length=None):
    """
        Pads a list of sequences to the same length
        Input:
            seqs: list of sequences
            batch_first: if True, output is (batch, seq_len, ...)
                            else, output is (seq_len, batch, ...)
            padding_value: value to pad with
    """
    max_len = max(len(seq) for seq in seqs)
    if max_length is not None:
        max_len = max_length
    if batch_first:
        padded_seqs = np.full((len(seqs), max_len, *seqs[0].shape[1:]),
                              padding_value,
                              dtype=seqs[0].dtype)
    else:
        padded_seqs = np.full((max_len, len(seqs), *seqs[0].shape[1:]),
                              padding_value,
                              dtype=seqs[0].dtype)

    for i, seq in enumerate(seqs):
        if batch_first:
            padded_seqs[i, :len(seq)] = seq
        else:
            padded_seqs[:len(seq), i] = seq

    return padded_seqs.astype(np.float32)


def create_window_seqs(x, y, min_sequence_length, weeks_ahead, pad_value):
    """
    Creates windows of fixed size with appended zeros
    Input:
        x: features [n_samples, n_features]
        y: targets, [n_samples, 1]
        min_sequence_length: minimum length of sequence
        weeks_ahead: number of weeks ahead to predict
        pad_value: value to pad with
    """
    seqs, mask_seqs, targets, mask_ys = [], [], [], []
    for idx in range(min_sequence_length, x.shape[0] + 1, 1):
        # Sequences
        seqs.append(x[:idx, :])
        # mask_seqs.append(np.ones(idx))
        mask_seqs.append(np.zeros(idx))

        # Targets
        y_val = y[idx:idx + weeks_ahead]
        y_ = np.ones((weeks_ahead, y_val.shape[1])) * pad_value
        y_[:y_val.shape[0], :] = y_val
        mask_y = np.zeros(weeks_ahead)
        mask_y[:len(y_val)] = 1
        targets.append(y_)
        mask_ys.append(mask_y)

    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask_seqs = pad_sequence(mask_seqs, batch_first=True, padding_value=-np.inf)
    ys = pad_sequence(targets, batch_first=True, padding_value=pad_value)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0)

    return seqs, mask_seqs, ys, mask_ys


def create_fixed_window_seqs(x, y, sequence_length, weeks_ahead, pad_value):
    seqs, mask_seqs, ys, mask_ys = [], [], [], []
    for idx in range(sequence_length, x.shape[0] + 1, 1):
        # Sequences
        seqs.append(x[idx-sequence_length:idx, :])
        mask_seqs.append(np.zeros(sequence_length, dtype=float))

        # Targets
        y_val = y[idx:idx + weeks_ahead]
        y_ = np.ones((weeks_ahead, y_val.shape[1])) * pad_value
        y_[:y_val.shape[0], :] = y_val
        mask_y = np.zeros(weeks_ahead)
        mask_y[:len(y_val)] = 1
        ys.append(y_)
        mask_ys.append(mask_y)
    seqs = np.array(seqs, dtype=float)
    mask_seqs = np.array(mask_seqs)
    ys = pad_sequence(ys, batch_first=True, padding_value=pad_value)
    mask_ys = pad_sequence(mask_ys, batch_first=True, padding_value=0)
    return seqs, mask_seqs, ys, mask_ys


def split_seqs(xs, mask_xs, ys, mask_ys, weeks_ahead, cal_num, test_num):
    """Split one sequences into train, calibration and test set."""
    total_num = len(xs)
    train_num = total_num - cal_num - test_num

    def split_seqs_helper(start, end):
        return xs[start:end], mask_xs[start:end], ys[start:end], mask_ys[start:end]

    trains = split_seqs_helper(0, train_num)
    cals = split_seqs_helper(train_num, train_num + cal_num)
    tests = split_seqs_helper(train_num + cal_num, train_num + cal_num + test_num)
    
    return trains, cals, tests


def split_seqs_for_cp(seqs, mask_seqs, ys, mask_ys, weeks_ahead, weeks4cal=3):
    cal_seqs = seqs[-(weeks4cal + weeks_ahead): -(weeks_ahead)]
    cal_mask_seqs = mask_seqs[-(weeks4cal + weeks_ahead): -(weeks_ahead)]
    cal_ys = ys[-(weeks4cal + weeks_ahead): -(weeks_ahead)]
    cal_mask_ys = mask_ys[-(weeks4cal + weeks_ahead): -(weeks_ahead)]
    
    train_seq_size = len(seqs) - weeks4cal - weeks_ahead

    return cal_seqs, cal_mask_seqs, cal_ys, cal_mask_ys, train_seq_size


def prepare_ds(xs, xs_masks, ys, ys_masks, regions, metas, test=False, return_ds=True, with_week_id=True):
    regions = np.array(regions, dtype="str").tolist()
    metas = np.concatenate(metas, axis=0)
    
    xs = np.concatenate(xs, axis=0)
    xs_masks = np.concatenate(xs_masks, axis=0)
    #  add a dimension for the number of features
    xs_masks = np.expand_dims(xs_masks, 2)
    
    if not test:
        ys = np.concatenate(ys, axis=0)
        ys_masks = np.concatenate(ys_masks, axis=0)
        ys_masks = np.expand_dims(ys_masks, 2)
        ys_masks = np.array(ys_masks)
    else:
        ys = np.ones((xs.shape[0], 2))
        ys_masks = np.ones((xs.shape[0], 2))

    if return_ds:
        dataset = SeqData(regions, metas, xs, xs_masks, ys, ys_masks, with_week_id)
        return dataset
    return xs, xs_masks, ys, ys_masks, regions, metas


class SeqData(torch.utils.data.Dataset):

    def __init__(self, region, metas, X, mask_X, y, mask_y, with_week_id=True):
        self.region = region
        self.metas = metas
        self.X = X
        self.mask_X = mask_X
        self.y = y
        self.mask_y = mask_y
        self.with_week_id = with_week_id

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.with_week_id:
            return (self.region[idx], self.metas[idx, :], self.X[idx, :, :-1], self.mask_X[idx], self.y[idx], self.mask_y[idx], self.X[idx, :, -1])
        return (self.region[idx], self.metas[idx, :], self.X[idx, :, :], self.mask_X[idx], self.y[idx], self.mask_y[idx], self.mask_y[idx])


class QuantileData(torch.utils.data.Dataset):
    def __init__(self, region, metas, embs, y_preds, residues, seq_lens, yqs):
        self.region = region
        self.metas = metas
        self.embs = embs
        self.yqs = yqs
        self.y_preds = y_preds
        self.residues = residues
        self.seq_lens = seq_lens

    def __len__(self):
        return self.yqs.shape[0]

    def __getitem__(self, idx):
        return (self.region[idx], self.metas[idx, :], self.embs[idx, :], self.y_preds[idx], self.residues[idx, :], self.seq_lens[idx], self.yqs[idx])


class CPData(torch.utils.data.Dataset):
    def __init__(self, xs, y_hats, region_ids, week_ids, ys):
        self.xs = xs
        self.y_hats = y_hats
        self.region_ids = region_ids
        self.week_ids = week_ids
        self.ys = ys
        self.qs = []
        for i in range(len(ys)):
            self.qs.append(np.abs(ys[i] - y_hats[i]))

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx, :, :], self.y_hats[idx], self.region_ids[idx], self.week_ids[idx], self.ys[idx], self.qs[idx]
