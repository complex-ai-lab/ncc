# Prepare data for conformal prediction. Train the base model using availiable data up until now.
import torch.nn as nn
import torch
from epiweeks import Week
import yaml
import numpy as np
import random
import argparse
from tqdm import tqdm
from datetime import timedelta

from forecaster.utils import ForecasterTrainer, EarlyStopping, pickle_save, decode_onehot, last_nonzero
from forecaster.seq2seq import Seq2seq
from forecaster.simplebasemodels import ThetaModelWrapper, RandomForestWrapper, ArimaWrapper
from forecaster.transformer import TransformerEncoderDecoder
from forecaster.informer2 import Informer2
from forecaster.dlinear import DlinearModel
from forecaster.load_covid import prepare_data, prepare_region_fine_tuning_data
from forecaster.load_power import prepare_power_data, convert_to_datetime, datetime2str, load_all_power_data
from forecaster.load_std_data import prepare_std_data, convert_from_str, convert_to_str, pred_delta_time, load_all_std_data

import warnings
warnings.filterwarnings("ignore")

###################
# add new dataset #
###################

def map_data_params_file(ot_params):
    data_params_file = '../../setup/covid_mortality.yaml'
    if ot_params['dataset'] == 'power':
        data_params_file = '../../setup/power.yaml'
    if ot_params['dataset'] == 'weather':
        data_params_file = '../../setup/weather.yaml'
    if ot_params['dataset'] == 'smd':
        data_params_file = '../../setup/smd.yaml'
    if ot_params['dataset'] == 'stock':
        data_params_file = '../../setup/stock.yaml'
    if ot_params['dataset'] == 'electricity':
        data_params_file = '../../setup/electricity.yaml'
    return data_params_file


def assign_last_train_time(params, last_train_time):
    dataset_name = params['dataset']
    if dataset_name == 'power':
        params['last_train_time'] = datetime2str(last_train_time)
    elif dataset_name == 'covid':
        params['last_train_time'] = Week.fromstring(last_train_time).cdcformat()
    else:
        params['last_train_time'] = convert_to_str(last_train_time, dataset=params['dataset'])
    return params


def prepare_dataloader_helper(params, power_df):
    if params['dataset'] == 'power':
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_power_data(power_df, params)
        x_dim += 1
    elif params['dataset'] == 'covid':
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_data(params)
    else:
        # uncomment to check if target idx and target match
        # power_df[params['target']] = -1000000
        train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_std_data(power_df, params)
        x_dim += 1
        # for batch in test_dataloader:
        #     regions, meta, x, x_mask, y, y_mask, weekid = batch
        #     print(x[:, :, params['target_idx']])
        #     exit(0)
    return train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length

###################
# add new dataset #
###################

def model_init(params, metas_dim, x_dim, device, seq_length):
    model = None
    if params['model_name'] == 'seq2seq':
        model = Seq2seq(
            metas_train_dim=metas_dim,
            x_train_dim=x_dim-1,
            device=device,
            weeks_ahead=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            out_layer_dim=params['model_parameters']['out_layer_dim'],
            out_dim=1
        )
    if params['model_name'] == 'transformer':
        model = TransformerEncoderDecoder(
            input_dim=x_dim-1,
            output_dim=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            seq_length=seq_length,
            num_layers=params['model_parameters']['num_layers'],
            num_heads=params['model_parameters']['num_heads'],
            num_regions=metas_dim,
            rnn_hidden_dim=params['model_parameters']['rnn_hidden_dim'],
            rnn_layers=params['model_parameters']['rnn_layers'],
        )
    if params['model_name'] == 'dlinear':
        model = DlinearModel(
            seq_len=seq_length,
            pred_len=params['weeks_ahead'],
            individual=True,
            enc_in=x_dim-1,
            target_idx=params['target_idx'],
        )
    if params['model_name'] == 'informer2':
        model = Informer2(
            enc_in=x_dim-1,
            dec_in=1,
            output_dim=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            seq_length=seq_length,
            # hidden_dim=params['model_parameters']['hidden_dim'],
            # seq_length=seq_length,
            # num_layers=params['model_parameters']['num_layers'],
            # num_heads=params['model_parameters']['num_heads'],
            # num_regions=metas_dim,
            rnn_hidden_dim=params['model_parameters']['rnn_hidden_dim'],
            rnn_layers=params['model_parameters']['rnn_layers'],
        )    
    if params['model_name'] == 'theta':
        model = ThetaModelWrapper(aheads=params['weeks_ahead'], period=params['theta_period'], target_idx=params['target_idx'])
    
    if params['model_name'] == 'randomforest':
        model = RandomForestWrapper(aheads=params['weeks_ahead'], target_idx=params['target_idx'])
    
    if params['model_name'] == 'arima':
        model = ArimaWrapper(aheads=params['weeks_ahead'], target_idx=params['target_idx'])
    return model


def region_fine_tuning(params, model_state_dict, target_region, all_dataloaders, seq_length):
    device = torch.device(params['device'])
    train_dataloader, val_dataloader, _, x_dim, _ = all_dataloaders[target_region]
    metas_dim = len(params['regions'])
    
    # load pretrained model states
    model = model_init(params, metas_dim, x_dim, device, seq_length)
    
    model = model.to(device)
    model.load_state_dict(model_state_dict)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

    # create loss function
    loss_fn = nn.MSELoss()

    # create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=params['training_parameters']['gamma'],
        verbose=False)

    # create early stopping
    early_stopping = EarlyStopping(
        patience=50, verbose=False)

    # create trainer
    trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

    # train model
    for epoch in range(params['rft_epochs']):
        trainer.train(train_dataloader, epoch)
        val_loss = trainer.evaluate(val_dataloader, epoch)
        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    
    model.load_state_dict(early_stopping.model_state_dict)
    return model


def forecast(model, model_name, test_dataloader, device, is_test, true_scale, ys_scalers):
    model.eval()
    predictions = {}
    addition_info = {}
    with torch.no_grad():
        for batch in test_dataloader:
            # get data
            regions, meta, x, x_mask, y, y_mask, weekid = batch
            x_mask = x_mask.type(torch.float)
            regionid = decode_onehot(meta)
            weekid = last_nonzero(weekid)
            if model_name == 'seq2seq':
                # send to device
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred, emb = model.forward(x, x_mask, meta, output_emb=True)
                y_pred = y_pred.cpu().numpy()
                emb = emb.cpu().numpy()
            
            if model_name == 'transformer':
                # send to device
                regionid = regionid.to(device)
                weekid = weekid.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred = model.forward(x, x_mask, regionid, weekid).unsqueeze(-1).cpu().numpy()
                emb = np.zeros(len(regions))
            
            if model_name == 'theta':
                y_pred = model.forward(x)[:, None]
                emb = np.zeros(len(regions))
            
            if model_name == 'dlinear':
                y_pred = model.forward(x).unsqueeze(-1).cpu().numpy()
                emb = np.zeros(len(regions))
            
            if model_name == 'informer2':
                regionid = regionid.to(device)
                weekid = weekid.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)
                y_pred = model.forward(x, x_mask, regionid, weekid).unsqueeze(-1).cpu().numpy()
            
            if model_name == 'randomforest':
                y_pred = model.forward(x)[:, None, :]
                emb = np.zeros(len(regions))
            
            if model_name == 'arima':
                y_pred = model.forward(x)[:, :, None]
                emb = np.zeros(len(regions))
                # print(y_pred.shape)
                # print(y.shape)
                # print(y_pred[0, 0, 0])
                # print(y[0, 0, 0])
                # exit(0)
            
            if is_test:
                y = np.zeros((len(regions), len(y_pred[0])))
            else:
                y = y.numpy()
                y = y[:, :, 0]
            meta = meta.cpu().numpy()

            # use scaler to inverse transform
            for i, region in enumerate(regions):
                if true_scale:
                    predictions[region] = ys_scalers[region].inverse_transform(y_pred[i]).reshape(-1)
                    y_in_true_scale = ys_scalers[region].inverse_transform(y[i].reshape(-1, 1)).reshape(-1)
                    addition_info[region] = (y_in_true_scale, y_mask[i], x[i], x_mask[i], weekid[i])
                else:
                    predictions[region] = y_pred[i].reshape(-1)
                    addition_info[region] = (y[i], y_mask[i], x[i], x_mask[i], weekid[i])
    return predictions, addition_info


def train_and_forcast(last_train_time, params, pretrained_model_state, train=True, power_df=None):
    params = assign_last_train_time(params, last_train_time)
    device = torch.device(params['device'])
    true_scale = params['true_scale']
    # decide if this is the test week
    is_test = False
    if params['test_time'] == params['last_train_time']:
        is_test = True
    
    train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_dataloader_helper(params, power_df)
    metas_dim = len(params['regions'])
    
    # if using a simple model, skip training
    skip_training = False
    if params['model_name'] == 'theta' or params['model_name'] == 'randomforest' or params['model_name'] == 'arima':
        skip_training = True

    # create model
    model = model_init(params, metas_dim, x_dim, device, seq_length)
    
    if params['model_name'] == 'randomforest':
        model.train(train_dataloader)

    if not skip_training:
        model = model.to(device)
            
        if train or pretrained_model_state is None:
            epochs = params['training_parameters']['epochs']
            if params['week_retrain'] == False and pretrained_model_state is not None:
                model.load_state_dict(pretrained_model_state)
                epochs = params['week_retrain_epochs']

            # create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

            # create loss function
            loss_fn = nn.MSELoss()

            # create scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=params['training_parameters']['gamma'],
                verbose=False)

            # create early stopping
            early_stopping = EarlyStopping(
                patience=100, verbose=False)

            # create trainer
            trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

            # train model
            for epoch in range(epochs):
                trainer.train(train_dataloader, epoch)
                val_loss = trainer.evaluate(val_dataloader, epoch)
                scheduler.step()
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break
            
            pretrained_model_state = early_stopping.model_state_dict

        model.load_state_dict(pretrained_model_state)
    
    rft_models = {}
    predictions = {}
    addition_info = {}
    
    # fine-tuning for each state
    if params['region_fine_tuning'] == True:
        all_dataloaders = prepare_region_fine_tuning_data(params)
        for region in params['regions']:
            rft_models[region] = region_fine_tuning(params, pretrained_model_state, region, all_dataloaders, seq_length)
        for region in params['regions']:
            rft_model = rft_models[region]
            _, _, region_test_dataloader, _, _ = all_dataloaders[region]
            region_predictions, region_addition_info = forecast(rft_model, params['model_name'], region_test_dataloader, device, is_test, true_scale, ys_scalers)
            predictions[region] = region_predictions[region]
            addition_info[region] = region_addition_info[region]
    else:
        predictions, addition_info = forecast(model, params['model_name'], test_dataloader, device, is_test, true_scale, ys_scalers)
    return predictions, addition_info, pretrained_model_state


def get_params(input_file='1'):
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            ot_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    data_params_file = '../../setup/covid_mortality.yaml'
    if ot_params['dataset']:
        data_params_file = map_data_params_file(ot_params)
    
    with open(data_params_file, 'r') as stream:
        try:
            task_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    with open('../../setup/seq2seq.yaml', 'r') as stream:
        try:
            model_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    # merge params
    params = {**task_params, **model_params}
    
    if 'training_params' in ot_params:
        for key, value in ot_params['training_params'].items():
            params['training_params'][key] = value
    if 'model_params' in ot_params:
        for key, value in ot_params['model_params'].items():
            params['model_params'][key] = value
    
    # overwrite using online training params
    for key, value in ot_params.items():
        if key == 'training_params' or key == 'model_params':
            continue
        params[key] = value

    if params['dataset'] == 'covid':
        params['data_params']['start_time'] = Week.fromstring(params['data_params']['start_time']).cdcformat()
        params['test_time'] = Week.fromstring(str(params['test_time'])).cdcformat()
    
    print('Paramaters loading success.')
    
    return params


def run_online_training(params):
    # TODO: update parameters in covid params to match power dataset
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    
    base_pred = []
    test_pred = None
    pretrained_model_state = None
    
    if params['dataset'] == 'power':
        starting_time = str(params['pred_starting_time'])
        test_time = str(params['test_time'])
        test_time = convert_to_datetime(test_time)
        total_steps = int(params['total_steps'])
        power_df = load_all_power_data()
        # get base model predictions
        train = False
        for i in tqdm(range(total_steps)):
            if i%params['week_retrain_period'] == 0:
                train = True
            else:
                train = False
            current_time = convert_to_datetime(starting_time) + timedelta(minutes=i)
            if current_time != test_time:
                predictions, addition_infos, pretrained_model_state = train_and_forcast(current_time, params, pretrained_model_state, power_df=power_df, train=train)
                base_pred.append((predictions, addition_infos))
            if current_time >= test_time:
                test_pred, _, _ = train_and_forcast(current_time, params, pretrained_model_state, power_df=power_df)
                break
    elif params['dataset'] == 'covid':
        starting_week = str(params['pred_starting_time'])
        test_week = str(params['test_time'])
        total_weeks_number = int(params['total_steps'])
        # get base model predictions
        for i in tqdm(range(total_weeks_number)):
            if i%params['week_retrain_period'] == 0:
                train = True
            else:
                train = False
            current_week = (Week.fromstring(starting_week) + i).cdcformat()
            if current_week != test_week:
                predictions, addition_infos, pretrained_model_state = train_and_forcast(current_week, params, pretrained_model_state, train=train)
                base_pred.append((predictions, addition_infos))
            if current_week == test_week:
                test_pred, _, _ = train_and_forcast(current_week, params, pretrained_model_state)
                break
    else:
        starting_time = str(params['pred_starting_time'])
        test_time = str(params['test_time'])
        test_time = convert_from_str(test_time, dataset=params['dataset'])
        total_steps = int(params['total_steps'])
        std_df = load_all_std_data(params)
        # get base model predictions
        train = False
        for i in tqdm(range(total_steps)):
            if i%params['week_retrain_period'] == 0:
                train = True
            else:
                train = False
            current_time = convert_from_str(starting_time, params['dataset']) + pred_delta_time(i, params['dataset'])
            if current_time != test_time:
                predictions, addition_infos, pretrained_model_state = train_and_forcast(current_time, params, pretrained_model_state, power_df=std_df, train=train)
                base_pred.append((predictions, addition_infos))
            if current_time >= test_time:
                test_pred, _, _ = train_and_forcast(current_time, params, pretrained_model_state, power_df=std_df)
                break
    
    results = {
        'params': params,
        'base_pred': base_pred,
        'test_pred': test_pred 
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    args = parser.parse_args()
    input_file = args.input    # for example: 1
    
    params = get_params(input_file)
    data_id = int(params['data_id'])
    results = run_online_training(params)
    pickle_save(f'../../results/base_pred/saved_pred_{data_id}.pickle', results)


if __name__ == '__main__':
    main()