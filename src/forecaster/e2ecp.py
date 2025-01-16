import torch
import random
import numpy as np
import argparse
from pathlib import Path

from forecaster.utils import pickle_load, pickle_save, load_yaml_params
from forecaster.e2ecp_utils import End2endCPTrainerForecaster, init_model, End2endCPDataset

SAVED_RESULTS_PATH = '/scratch/alrodri_root/alrodri0/liruipu/saved_results'
SERVER_PATH = '/scratch/alrodri_root/alrodri0/liruipu/base_pred'
LOCAL_PATH = '../../results/base_pred'


def e2ecp_train(e2ecp_params, exp_id):
    training_params = e2ecp_params['training_params']
    model_params = e2ecp_params['model_params']
    print(training_params)
    
    week_ahead_list = e2ecp_params['week_ahead_list']
    alphas = e2ecp_params['alphas']
    
    seed = e2ecp_params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # load e2ecp data
    data_file_id = e2ecp_params['data_file_id']
    all_base_preds = None
    if e2ecp_params['greatlakes']:
        all_base_preds = pickle_load(f'{SERVER_PATH}/saved_pred_{data_file_id}.pickle', version5=True)
    else:
        all_base_preds = pickle_load(f'{LOCAL_PATH}/saved_pred_{data_file_id}.pickle', version5=True)
    base_pred_ds = 'covid'
    if 'dataset' in all_base_preds['params']:
        base_pred_ds = all_base_preds['params']['dataset']
    base_pred = all_base_preds['base_pred']
    e2ecp_dataset = End2endCPDataset(base_pred, week_ahead_list, base_pred_ds)
    
    # prepare model
    data, scalar = e2ecp_dataset.e2ecp_seq_data(training_params)
    regions = e2ecp_dataset.get_regions()
    x_dim = e2ecp_dataset.get_x_dim()
    seq_length = e2ecp_dataset.get_seq_length()
    
    with_week_id = True if base_pred_ds == 'covid' else False
    
    model = init_model(model_params, seq_length, x_dim, len(regions), alphas, with_week_id, training_params['cov_window_size'], 4)
    if e2ecp_params['use_pretrained_model']:
        model.load_state_dict(torch.load(Path(e2ecp_params['pretrained_model_path'])))

    if e2ecp_params['train']:
        training_params['retrain_window'] = -1
        e2ecp_trainerforecaster = End2endCPTrainerForecaster(model, regions, week_ahead_list, alphas, scalar=scalar, dataset=base_pred_ds, cf_lr=model_params['cf_lr'], tta_reg_factor=training_params['tta_reg_factor'], tta_lr=training_params['tta_lr'])
        model, losses = e2ecp_trainerforecaster.stage_train(training_params, data, regions)        
        pickle_save(Path(e2ecp_params['model_path'])/f'losses_{exp_id}.pkl', losses)
        torch.save(model.state_dict(), Path(e2ecp_params['model_path'])/f'{exp_id}.pt')
        if training_params['region_fine_tuning']:
            rf_models_path = Path(f'../../results/rf_{exp_id}')
            if not rf_models_path.exists():
                rf_models_path.mkdir()
            for region in regions:
                region_model = init_model(model_params, seq_length, x_dim, len(regions), alphas, with_week_id, training_params['cov_window_size'], 4)
                region_model.load_state_dict(model.state_dict())
                region_model, _ = e2ecp_trainerforecaster.stage_train(training_params, data, [region], pretrained_model=region_model)
                torch.save(region_model.state_dict(), rf_models_path/f'{exp_id}_{region}.pt')

    else:
        if training_params['retrain_with_cf']:
            model.params['e2ecf'] = True
        model.load_state_dict(torch.load(Path(e2ecp_params['model_path'])/f'{exp_id}.pt'))
        
        e2ecp_trainerforecaster = End2endCPTrainerForecaster(model, regions, week_ahead_list, alphas, scalar=scalar, dataset=base_pred_ds)
        
        if e2ecp_params['test_cp_lr_corrected']:
            e2ecp_trainerforecaster = End2endCPTrainerForecaster(model, regions, week_ahead_list, alphas, scalar=scalar, dataset=base_pred_ds, cf_lr=model_params['cf_lr'], tta_reg_factor=training_params['tta_reg_factor'], tta_lr=training_params['tta_lr'])

        # forecast
        all_predictions = e2ecp_trainerforecaster.forecast4eval(training_params, data)
        # save predictions
        pickle_save(f'../../results/{exp_id}.pkl', all_predictions)
        # region fine tuning predictions
        if training_params['region_fine_tuning']:
            rf_models_path = Path(f'../../results/rf_{exp_id}')
            all_rft_predictions = {}
            for region in regions:
                model = init_model(model_params, seq_length, x_dim, len(regions), alphas, with_week_id, training_params['cov_window_size'], 4)
                model.load_state_dict(torch.load(rf_models_path/f'{exp_id}_{region}.pt'))
                e2ecp_trainerforecaster = End2endCPTrainerForecaster(model, [region], week_ahead_list, alphas, scalar=scalar, dataset=base_pred_ds, cf_lr=model_params['cf_lr'], tta_reg_factor=training_params['tta_reg_factor'], tta_lr=training_params['tta_lr'])
                all_predictions_in_region = e2ecp_trainerforecaster.forecast4eval(training_params, data)
                all_rft_predictions[region] = all_predictions_in_region[region]
            pickle_save(f'../../results/{exp_id}_rfted.pkl', all_rft_predictions)


def load_all_params(input_file):
    # load yaml files
    e2ecp_params = load_yaml_params('../../setup/e2ecp.yaml')
    # load exp params
    e2ecp_exp_params = load_yaml_params(f'../../setup/cp_exp_params/{input_file}.yaml')
    # rewrite
    for key, value in e2ecp_exp_params.items():
        if key != 'training_params' and key != 'model_params':
            # print(key)
            e2ecp_params[key] = value
    if 'training_params' in e2ecp_exp_params:
        for key, value in e2ecp_exp_params['training_params'].items():
            e2ecp_params['training_params'][key] = value
    if 'model_params' in e2ecp_exp_params:
        for key, value in e2ecp_exp_params['model_params'].items():
            e2ecp_params['model_params'][key] = value
    return e2ecp_params


def main():
    # week_ahead_list = [1]
    # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--greatlakes', '-g', action='store_true')
    args = parser.parse_args()
    input_file = args.input    # for example: 1
    exp_id = input_file
    e2ecp_params = load_all_params(input_file)
    e2ecp_params['train'] = args.train
    e2ecp_params['greatlakes'] = args.greatlakes
    
    e2ecp_train(e2ecp_params, f'{exp_id}')


if __name__ == '__main__':
    main()
    

    
                
            
    