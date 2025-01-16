# TODO: now using training loss as val loss

import torch
import numpy as np

from forecaster.utils import EarlyStopping
from forecaster.loss_utils import LossCalculator, LossTracker, MonotonicityLoss
from forecaster.end2endmodel import TransformerCP
from forecaster.test_time_adapt import monotonicity_measure_func1, get_hs_mask


def region2idx(regions, target_region):
    for i, region in enumerate(regions):
        if region == target_region:
            return i
    assert False


def mean_torch_loss(losses):
    mean_loss = None
    for loss in losses:
        if mean_loss is None:
            mean_loss = loss
        else:
            mean_loss += loss
    return mean_loss / len(losses)


class End2endCPDataset:
    def __init__(self, base_pred, week_ahead_list, dataset='covid'):
        self.base_pred = base_pred
        self.week_ahead_list = week_ahead_list
        self.dataset = dataset

    def e2ecp_seq_data(self, training_params):
        # week ahead : [1, 2, 3, 4]. When saving data to dictionarys, use week ahead as keys
        # week ahead idx : [0, 1, 2, 3]. When indexing predictions, use week ahead idx.
        self.seq_data = {}
        self.regions = None
        self.x_dim = None
        self.seq_len = None
        for saved_info in self.base_pred:
            predictions, addition_infos = saved_info
            if self.regions is None:
                # init struct
                self.regions = list(predictions.keys())
                for region in self.regions:
                    self.seq_data[region] = {}
                    for week_ahead in self.week_ahead_list:
                        self.seq_data[region][week_ahead] = []
            for region in self.regions:
                for week_ahead in self.week_ahead_list:
                    week_ahead_idx = week_ahead - 1
                    prediction = predictions[region]
                    addition_info = addition_infos[region]    
                    y, y_mask, x, x_mask, week_id = addition_info
                    if self.dataset != 'covid':
                        week_id = 0
                    if self.x_dim is None:
                        self.x_dim = x.shape[-1]
                        self.seq_len = x.shape[-2]
                    # x, y_hat, region_id, week_id, week_ahead_id, y
                    current_data_point = (x, prediction[week_ahead_idx], region2idx(self.regions, region), week_id, week_ahead, y[week_ahead_idx])
                    self.seq_data[region][week_ahead].append(current_data_point)
        # normalize
        scalar = E2ecpScalar(self.regions, self.week_ahead_list)
        data_size = len(self.seq_data[self.regions[0]][self.week_ahead_list[0]])
        training_steps = data_size - training_params['val_weeks'] - training_params['test_weeks']
        for region in self.regions:
            for week_ahead in self.week_ahead_list:
                ys = []
                for t in range(training_steps):
                    ys.append(self.seq_data[region][week_ahead][t][5])
                ys = np.array(ys)
                scalar.fit(ys, region, week_ahead, use_scalar=training_params['use_scalar'])
        for region in self.regions:
            for week_ahead in self.week_ahead_list:
                for i in range(data_size):
                    x, y_hat, region_id, week_id, week_ahead_id, y = self.seq_data[region][week_ahead][i]
                    y_hat = scalar.transform(y_hat, region, week_ahead)
                    y = scalar.transform(y, region, week_ahead)
                    self.seq_data[region][week_ahead][i] = (x, y_hat, region_id, week_id, week_ahead_id, y)
        return self.seq_data, scalar

    def get_regions(self):
        return self.regions
    
    def get_x_dim(self):
        return self.x_dim
    
    def get_seq_length(self):
        return self.seq_len


class E2ecpScalar:
    # vars means stds in this class
    def __init__(self, regions, week_ahead_list):
        self.regions = regions
        self.week_ahead_list = week_ahead_list
        self.vars = {}
        self.means = {}
        for region in self.regions:
            self.vars[region] = {}
            self.means[region] = {}
    
    def fit(self, data, region, week_ahead, use_scalar=True):
        """
        Data: np.array without nan vals.
        """
        if use_scalar:
            self.vars[region][week_ahead] = np.std(data)
            self.means[region][week_ahead] = np.mean(data)
        else:
            self.vars[region][week_ahead] = 1
            self.means[region][week_ahead] = 0
        
    def transform(self, data, region, week_ahead):
        data = data - self.means[region][week_ahead]
        data = data / self.vars[region][week_ahead]
        return data
    
    def inv_tranform(self, data, region, week_ahead, is_score=False):
        data = data * self.vars[region][week_ahead]
        if not is_score:
            data = data + self.means[region][week_ahead]
        return data


def init_model(params, seq_length, x_dim, num_regions, alphas, with_week_id, cov_window_size, num_aheads):
    if params['model_name'] == 'transformer':
        # update params
        params['with_week_id'] = with_week_id
        params['alphas'] = alphas
        params['num_regions'] = num_regions
        params['num_aheads'] = num_aheads
        params['window_size'] = cov_window_size
        params['encoder_input_dim'] = x_dim
        params['seq_length'] = seq_length
        model = TransformerCP(params)
    return model


def format_data(data, device):
    """
    Data is a tuple of (x, y_hat, region_id, week_id, week_ahead_id, y).
    If x.shape has a length of 2, expand it to 3.
    If any of the entries are not torch.Tensor, change to tensor.
    week id is a number.
    """
    x, y_hat, region_id, week_id, week_ahead_id, y = data
    if len(x.shape) == 2:
        x = torch.Tensor(x[None, :, :]).to(device)
        y_hat = torch.Tensor([y_hat])[None, :].to(device)
        region_id = torch.Tensor([region_id])[None, :].int().to(device)
        week_id = torch.Tensor([week_id])[None, :].int().to(device)
        week_ahead_id = torch.Tensor([week_ahead_id])[None, :].int().to(device)
        y = torch.Tensor([y])[None, :].to(device)
    return x, y_hat, region_id, week_id, week_ahead_id, y


def prepare_batch_input(data, regions, t, cov_window_size, week_ahead, device, alphas, error_seqs, score_seqs, pg_seqs, q_adjs):
    """_summary_

    Args:
        data (_type_): _description_
        regions (_type_): _description_
        t (_type_): _description_
        cov_window_size (_type_): _description_
        week_ahead (_type_): _description_
        device (_type_): _description_
        alphas (_type_): _description_
        error_seqs (_type_): in each region: (alphas x time steps)
        score_seqs (_type_): in each region: (alphas x time steps x 2)
        pg_seqs (_type_): in each region: (time steps x 2)
        q_adjs: in each region: (alphas)

    Returns:
        (batch inputs, shapes in a batch):
        xs: (x sequence length x x dim)
        y_hats: (x 1)
        ys: (x 1)
        region_ids: (x 1)
        week_ids: (x 1)
        week_ahead_ids: (x 1)
        scores: (x 1)
        input_alphases: (x alphas)
        region_idxs: (regions)
        cur_error_seqs: (seq x alphas)
        cur_score_seqs: (seq x alphas x 2)
        cur_pg_seqs: (seq x 2)
        cur_q_adjs: (x alphas)
    """
    num_regions = len(regions)
    region_idxs = np.arange(num_regions)
    np.random.shuffle(region_idxs)
    xs, y_hats, region_ids, week_ids, week_ahead_ids, ys, input_alphases, cur_error_seqs, cur_score_seqs, cur_pg_seqs, cur_q_adjs = [], [], [], [], [], [], [], [], [], [], []
    for i in range(num_regions):
        region_idx = region_idxs[i]
        region = regions[region_idx]
        # get alphas, error_seqs, score_seqs
        cur_error_seq = error_seqs[region][:, t:t+cov_window_size]
        cur_error_seq = cur_error_seq[None, :, :].permute(0, 2, 1).to(device)
        cur_score_seq = score_seqs[region][:, t:t+cov_window_size, :]
        cur_score_seq = cur_score_seq[None, :, :].permute(0, 2, 1, 3).to(device)
        cur_pg_seq = pg_seqs[region][t:t+cov_window_size, :]
        cur_pg_seq = cur_pg_seq[None, :, :].to(device)
        input_alphas = torch.tensor(alphas)[None, :].to(device)
        # get other input features
        x, y_hat, region_id, week_id, week_ahead_id, y = format_data(data[region][week_ahead][t], device=device)
        # append
        xs.append(x)
        y_hats.append(y_hat)
        ys.append(y)
        region_ids.append(region_id)
        week_ids.append(week_id)
        week_ahead_ids.append(week_ahead_id)
        cur_error_seqs.append(cur_error_seq)
        cur_score_seqs.append(cur_score_seq)
        cur_pg_seqs.append(cur_pg_seq)
        input_alphases.append(input_alphas)
        cur_q_adjs.append(torch.tensor(q_adjs[region])[None, :].to(device))
    xs = torch.cat(xs, dim=0)
    y_hats = torch.cat(y_hats, dim=0)
    ys = torch.cat(ys, dim=0)
    scores = torch.abs(y_hats - ys)
    region_ids = torch.cat(region_ids, dim=0)
    week_ids = torch.cat(week_ids, dim=0)
    week_ahead_ids = torch.cat(week_ahead_ids, dim=0)
    cur_error_seqs = torch.cat(cur_error_seqs, dim=0)
    cur_score_seqs = torch.cat(cur_score_seqs, dim=0)
    cur_pg_seqs = torch.cat(cur_pg_seqs, dim=0)
    input_alphases = torch.cat(input_alphases, dim=0)
    cur_qhat_seqs = cur_score_seqs[:, :, :, 1]
    tmp_score_seqs = cur_score_seqs[:, :, 0, 0][:, :, None]
    cur_score_seqs = torch.cat([cur_pg_seqs, tmp_score_seqs], dim=2)
    cur_q_adjs = torch.cat(cur_q_adjs, dim=0)
    return xs, y_hats, ys, region_ids, week_ids, week_ahead_ids, scores, input_alphases, region_idxs, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, cur_q_adjs


def update_batch_qts(idx2update, cur_q_hats, cur_ys, cur_yhats, scores, q_hats, error_seqs, score_seqs, pg_seqs, regions, region_idxs, alphas, q_adjs, updated_q_adjs):
    """
    cur_q_hats: (regions, alphas)
    scores: (regions)
    cur_ys: (regions, 1)
    cur_yhats: (regions, 1)
    
    Sequences to be updated:
    q_hats: in each region: (alphas x time steps)
    error_seqs: in each region: (alphas x time steps)
    score_seqs: in each region: (alphas x time steps x 2), 0 is scores, 1 is qhat
    pg_seqs: in each region: (time steps x 2), 0 is prediction, 1 is ground truth
    """
    for r in range(len(regions)):
        region = regions[region_idxs[r]]
        pg_seqs[region][idx2update, 0] = cur_yhats[r, 0]
        pg_seqs[region][idx2update, 1] = cur_ys[r, 0]
        q_adjs[region] = updated_q_adjs[r]
        for i in range(len(alphas)):
            q_hats[region][i, idx2update] = cur_q_hats[r, i]
            error_seqs[region][i, idx2update] = int(cur_q_hats[r, i] < scores[r])
            score_seqs[region][i, idx2update, 0] = scores[r]
            score_seqs[region][i, idx2update, 0] = scores[r]
    return q_hats, error_seqs, score_seqs, pg_seqs, q_adjs


class End2endCPTrainerForecaster:
    def __init__(self, model:TransformerCP, regions, week_ahead_list, alphas, scalar:E2ecpScalar, dataset='covid', cf_lr=0.01, tta_reg_factor=1, tta_lr=5e-4):
        self.model = model
        self.regions = regions
        self.week_ahead_list = week_ahead_list
        self.alphas = alphas
        self.scalar = scalar
        self.dataset = dataset
        self.cf_lr = cf_lr
        
        # tta
        self.tta_reg_factor = tta_reg_factor
        self.tta_lr = tta_lr
        self.mono_loss = MonotonicityLoss()
    
    
    def tta_forward_pass_no_FNN(self, data, regions2train, t, cov_window_size, week_ahead, device, model, pg_seqs, error_seqs, score_seqs, q_hats, q_adjs, monotonicity_threshold, monotonicity_score_func):
        xs, y_hats, ys, region_ids, week_ids, week_ahead_ids, scores, input_alphases, region_idxs, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, cur_q_adjs = prepare_batch_input(
            data=data,
            regions=regions2train,
            t=t,
            cov_window_size=cov_window_size,
            week_ahead=week_ahead,
            device=device,
            alphas=self.alphas,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            q_adjs=q_adjs,
        )
        # output shape: (batch size x num_alphas)
        # forward
        cur_q_hats, updated_q_adjs = model.forward(xs, region_ids, week_ids, week_ahead_ids, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, input_alphases, learning_rate=self.cf_lr, q_adj=cur_q_adjs, t=t)
        prev_qhat = cur_q_hats
        # print(prev_qhat)
        # print('TTA (FNN) print')
        # if monotonicity_score_func(cur_q_hats) <= monotonicity_threshold:
        #     print('TTA skipped')
        if monotonicity_score_func(cur_q_hats) > monotonicity_threshold:
            # print('TTA running')
            # print(monotonicity_score_func(cur_q_hats))
            # print(cur_q_hats)
            # init hs and optimizer
            hs = torch.zeros_like(cur_q_hats, requires_grad=True).to(cur_q_hats.device)
            cur_q_hats = cur_q_hats.detach()
            optimizer = torch.optim.SGD([hs], lr=self.tta_lr, weight_decay=1e-3)
            count = 0
            updated_q_hats = None
            while count < 200:
                count += 1
                updated_q_hats = cur_q_hats + hs
                if count%10==0 and monotonicity_score_func(updated_q_hats) < monotonicity_threshold:
                    # print('tta finished')
                    # print(hs)
                    # print(cur_q_hats)
                    # print(updated_q_hats)
                    break
                # calculate loss and backprop
                optimizer.zero_grad()
                loss_m = self.mono_loss(updated_q_hats, input_alphases)
                loss_reg = hs.abs().sum().mean()
                tta_loss = loss_m + self.tta_reg_factor * loss_reg
                # tta_loss = loss_m
                # print(f'loss m is {loss_m}, loss reg is {loss_reg * self.tta_reg_factor}')
                tta_loss.backward()
                optimizer.step()
                # print(hs)
            cur_q_hats = updated_q_hats
            # print('hs: ', hs)
            # print('qhats: ', cur_q_hats)
            # print('initial qhats: ', prev_qhat)
            # print('TTA finished')
        # update q_hats and Ets
        q_hats, error_seqs, score_seqs, pg_seqs, q_adjs = update_batch_qts(
            idx2update=t+cov_window_size+week_ahead-1,
            cur_q_hats=cur_q_hats,
            cur_ys=ys,
            cur_yhats=y_hats,
            scores=scores,
            q_hats=q_hats,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            regions=regions2train,
            region_idxs=region_idxs,
            alphas=self.alphas,
            q_adjs=q_adjs, 
            updated_q_adjs=updated_q_adjs
        )
        # unfreeze parameters of model
        for param in model.parameters():
            param.requires_grad = True
        return q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs
        
    
    def tta_forward_pass(self, data, regions2train, t, cov_window_size, week_ahead, device, model, pg_seqs, error_seqs, score_seqs, q_hats, q_adjs, monotonicity_threshold, monotonicity_score_func, mask=False, l1_reg=False):
        xs, y_hats, ys, region_ids, week_ids, week_ahead_ids, scores, input_alphases, region_idxs, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, cur_q_adjs = prepare_batch_input(
            data=data,
            regions=regions2train,
            t=t,
            cov_window_size=cov_window_size,
            week_ahead=week_ahead,
            device=device,
            alphas=self.alphas,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            q_adjs=q_adjs,
        )
        # output shape: (batch size x num_alphas)
        cur_q_hats, updated_q_adjs, prev_qhat = None, None, None
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tta_lr)
        # freeze parameters of model except tta model
        for param in model.parameters():
            param.requires_grad = False
        for param in model.tta_model.parameters():
            param.requires_grad = True
        cur_q_hats, updated_q_adjs, hs = model.tta_forward(xs, region_ids, week_ids, week_ahead_ids, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, input_alphases, learning_rate=self.cf_lr, q_adj=cur_q_adjs, t=t)
        prev_qhat = cur_q_hats
        # print('aaa')
        # print(monotonicity_score_func(cur_q_hats))
        # print('TTA (FNN) print')
        # if monotonicity_score_func(cur_q_hats) <= monotonicity_threshold:
        #     print('TTA skipped')
        if monotonicity_score_func(cur_q_hats) > monotonicity_threshold:
            # print('TTA running')
            # print(monotonicity_score_func(cur_q_hats))
            # print(cur_q_hats)
            count = 0
            while count < 200:
                count+=1
                cur_q_hats, updated_q_adjs, hs = model.tta_forward(xs, region_ids, week_ids, week_ahead_ids, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, input_alphases, learning_rate=self.cf_lr, q_adj=cur_q_adjs, t=t)
                # if prev_qhat is not None:
                #     print('Test q hat is not changing')
                #     assert torch.mean(prev_qhat - cur_q_hats) == 0
                cur_q_hats = cur_q_hats + hs
                # if count%10==0:
                #     print(monotonicity_score_func(cur_q_hats))
                if count%10==0 and monotonicity_score_func(cur_q_hats) < monotonicity_threshold:
                    # print(prev_qhat)
                    # print(hs)
                    # print(cur_q_hats)
                    break
                optimizer.zero_grad()
                loss_m = self.mono_loss(cur_q_hats, input_alphases)
                if l1_reg:
                    loss_reg = hs.abs().mean()
                else:
                    loss_reg = hs.pow(2).mean().sqrt()
                tta_loss = loss_m + loss_reg * self.tta_reg_factor
                # tta_loss = loss_m
                # print(f'loss m is {loss_m}, loss reg is {loss_reg * self.tta_reg_factor}')
                tta_loss.backward()
                optimizer.step()
            if mask:
                cur_q_hats, updated_q_adjs, hs = model.tta_forward(xs, region_ids, week_ids, week_ahead_ids, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, input_alphases, learning_rate=self.cf_lr, q_adj=cur_q_adjs, t=t)
                hs_mask = get_hs_mask(cur_q_hats)
                hs = hs * hs_mask
                cur_q_hats += hs
            
            # print('hs: ', hs)
            # print('qhats: ', cur_q_hats)
            # print('initial qhats: ', prev_qhat)
            # print('TTA finished')
            
        # update q_hats and Ets
        q_hats, error_seqs, score_seqs, pg_seqs, q_adjs = update_batch_qts(
            idx2update=t+cov_window_size+week_ahead-1,
            cur_q_hats=cur_q_hats,
            cur_ys=ys,
            cur_yhats=y_hats,
            scores=scores,
            q_hats=q_hats,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            regions=regions2train,
            region_idxs=region_idxs,
            alphas=self.alphas,
            q_adjs=q_adjs, 
            updated_q_adjs=updated_q_adjs
        )
        # unfreeze parameters of model
        for param in model.parameters():
            param.requires_grad = True
        return q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs
    
    def forward_pass(self, data, regions2train, t, cov_window_size, week_ahead, device, model, loss_calculator, alternate_training, epoch, pg_seqs, error_seqs, score_seqs, q_hats, q_adjs, eval=False):
        # shape: (num_alphas, cov_window_size+data_size)
        # data at i is put at pos regionidx[i]
        xs, y_hats, ys, region_ids, week_ids, week_ahead_ids, scores, input_alphases, region_idxs, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, cur_q_adjs = prepare_batch_input(
            data=data,
            regions=regions2train,
            t=t,
            cov_window_size=cov_window_size,
            week_ahead=week_ahead,
            device=device,
            alphas=self.alphas,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            q_adjs=q_adjs,
        )
        # output shape: (batch size x num_alphas)
        cur_q_hats, updated_q_adjs = model(xs, region_ids, week_ids, week_ahead_ids, cur_error_seqs, cur_score_seqs, cur_qhat_seqs, input_alphases, learning_rate=self.cf_lr, q_adj=cur_q_adjs)
        # update q_hats and Ets
        q_hats, error_seqs, score_seqs, pg_seqs, q_adjs = update_batch_qts(
            idx2update=t+cov_window_size+week_ahead-1,
            cur_q_hats=cur_q_hats,
            cur_ys=ys,
            cur_yhats=y_hats,
            scores=scores,
            q_hats=q_hats,
            error_seqs=error_seqs,
            score_seqs=score_seqs,
            pg_seqs=pg_seqs,
            regions=regions2train,
            region_idxs=region_idxs,
            alphas=self.alphas,
            q_adjs=q_adjs, 
            updated_q_adjs=updated_q_adjs
        )
        if eval:
            return q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs
        # loss
        loss, qloss, closs, eloss, mloss = loss_calculator.cal_loss(cur_q_hats, cur_error_seqs, input_alphases, scores, return_all_losses=True, alter_training=alternate_training, epoch=epoch)
        return q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, loss, qloss, closs, eloss, mloss
        
    def train(self, training_params, data, regions2train, pretrained_model=None, alternate_training=False, save_weights=True):
        device = torch.device(training_params['device'])
        model = pretrained_model
        if pretrained_model is None:
            # print('111')
            model = self.model
        # print('222')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['lr'])
        loss_calculator = LossCalculator(training_params, self.alphas)
        loss_tracker = LossTracker()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=training_params['gamma'], verbose=False)
        early_stopping = EarlyStopping(patience=50, verbose=False)

        # train model (from train_starting_week to total_weeks - val_weeks - test_weeks)
        for epoch in range(training_params['epochs']):
            # init
            data_size = len(data[self.regions[0]][self.week_ahead_list[0]])
            num_train_weeks = data_size - training_params['val_weeks'] - training_params['test_weeks']
            
            train_start_week = 0
            if training_params['retrain_window'] != -1:
                train_start_week = num_train_weeks - training_params['retrain_window']
            
            cov_window_size = training_params['cov_window_size']
            loss_tracker.epoch_start()
            for week_ahead in self.week_ahead_list:
                q_hats = {}
                error_seqs = {}
                pg_seqs = {}
                score_seqs = {}
                q_adjs = {}
                for region in regions2train:
                    q_hats[region] = torch.zeros((len(self.alphas), cov_window_size + data_size + week_ahead - 1))
                    error_seqs[region] = torch.ones((len(self.alphas), cov_window_size + data_size + week_ahead - 1))
                    
                    if training_params['init_err_seq_w_alpha'] == True:
                        for a in range(len(self.alphas)):
                            error_seqs[region][a] = error_seqs[region][a] * self.alphas[a]
                        
                    pg_seqs[region] = torch.zeros((cov_window_size + data_size + week_ahead - 1, 2))
                    score_seqs[region] = torch.zeros((len(self.alphas), cov_window_size + data_size + week_ahead - 1, 2))
                    q_adjs[region] = torch.zeros((len(self.alphas)))
                # train
                model.train()
                for t in range(num_train_weeks):
                    if t < train_start_week:
                        continue
                    q_hats, q_adjs, _, error_seqs, score_seqs, pg_seqs, loss, qloss, closs, eloss, mloss = self.forward_pass(
                        data=data,
                        regions2train=regions2train,
                        t=t,
                        cov_window_size=cov_window_size,
                        week_ahead=week_ahead,
                        device=device,
                        model=model,
                        loss_calculator=loss_calculator,
                        alternate_training=alternate_training,
                        epoch=epoch,
                        pg_seqs=pg_seqs,
                        error_seqs=error_seqs,
                        score_seqs=score_seqs,
                        q_hats=q_hats,
                        q_adjs=q_adjs,
                    )
                    # print(closs)
                    # print(qloss, closs, eloss, mloss)
                    loss_tracker.add_train_loss((qloss, closs, eloss, mloss))
                    loss_tracker.add_val_loss((qloss, closs, eloss, mloss))
                    # backward pass
                    optimizer.zero_grad()
                    try:
                        loss.backward()
                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
                        optimizer.step()
                    except:
                        print('Infinite loss, skipped!')
                # eval
                model.eval()
                for t_ in range(training_params['val_weeks']):
                    t = t_ + num_train_weeks
                    # prepare batch data
                    # data at i is put at pos regionidx[i]
                    q_hats, q_adjs,  _, error_seqs, score_seqs, pg_seqs, loss, qloss, closs, eloss, mloss = self.forward_pass(
                        data=data,
                        regions2train=regions2train,
                        t=t,
                        cov_window_size=cov_window_size,
                        week_ahead=week_ahead,
                        device=device,
                        model=model,
                        loss_calculator=loss_calculator,
                        alternate_training=alternate_training,
                        epoch=epoch,
                        pg_seqs=pg_seqs,
                        error_seqs=error_seqs,
                        score_seqs=score_seqs,
                        q_hats=q_hats,
                        q_adjs=q_adjs,
                    )
                    # loss_tracker.add_val_loss((qloss, closs, eloss, mloss))
            val_loss = loss_tracker.epoch_end(verbose=True)
            scheduler.step()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        saved_losses = loss_tracker.get_losses()
        if training_params['epochs'] > 0:
            model.load_state_dict(early_stopping.model_state_dict)
        if save_weights:
            self.model = model
        return self.model, saved_losses
    
    def stage_train(self, training_params, data, regions2train, pretrained_model=None, retrain=False, save_weights=True):
        if training_params['three_stage_training']:
            # save params
            num_stages = len(training_params['retrain_epochs'])
            epochs = training_params['epochs']
            qloss_factor = training_params['qloss_factor']
            closs_factor = training_params['closs_factor']
            eloss_factor = training_params['eloss_factor']
            # train
            model = pretrained_model
            for i in range(num_stages):
                alter_train = False if i < num_stages-1 else training_params['alter_training']
                training_params['epochs'] = int(training_params['epochs_in_stage'][i])
                training_params['qloss_factor'] = training_params['loss_factors'][i][0]
                training_params['closs_factor'] = training_params['loss_factors'][i][1]
                training_params['eloss_factor'] = training_params['loss_factors'][i][2]
                model, losses = self.train(training_params, data, regions2train, pretrained_model=model, save_weights=save_weights, alternate_training=alter_train)
            # restore params
            training_params['epochs'] = epochs
            training_params['qloss_factor'] = qloss_factor
            training_params['closs_factor'] = closs_factor
            training_params['eloss_factor'] = eloss_factor
        else:
            model, losses = self.train(training_params, data, regions2train, pretrained_model=pretrained_model, save_weights=save_weights)
        if save_weights:
            self.model = model
        return model, losses
    
    def retrain_model(self, training_params, data, model, steps_after_last_val):
        # update training params
        print('retrain started')
        saved_test_weeks = training_params['test_weeks']
        saved_epochs = training_params['epochs']
        saved_epochs_in_stage = training_params['epochs_in_stage']
        new_test_weeks = saved_test_weeks - steps_after_last_val
        training_params['test_weeks'] = new_test_weeks
        training_params['epochs'] = training_params['retrain_epoch']
        training_params['epochs_in_stage'] = training_params['retrain_epochs']
        model, _ = self.stage_train(training_params, data, self.regions, pretrained_model=model, save_weights=False)
        # reload training params
        training_params['test_weeks'] = saved_test_weeks
        training_params['epochs'] = saved_epochs
        training_params['epochs_in_stage'] = saved_epochs_in_stage
        print('retrain finished')
        return model
    
    def forecast4eval(self, training_params, data):
        """
        data is in the same format as training. The former part is used for generating errors (Ets). Only returns the forecasting of the testing weeks. So data can be the test weeks + a few weeks before the first test week.
        """
        use_tta = training_params['use_tta']
        use_tta_ffn = training_params['use_tta_ffn']
        device = torch.device(training_params['device'])
        cov_window_size = training_params['cov_window_size']
        data_size = len(data[self.regions[0]][self.week_ahead_list[0]])
        test_weeks = training_params['test_weeks']
        model = self.model.to(device)
        model.eval()
    
        all_predictions = {}
        for region in self.regions:
            all_predictions[region] = {}
            for week_ahead in self.week_ahead_list:
                all_predictions[region][week_ahead] = []
            
        for week_ahead in self.week_ahead_list:
            # shape: (num_alphas, cov_window_size+data_size)
            q_hats = {}
            error_seqs = {}
            pg_seqs = {}
            score_seqs = {}
            q_adjs = {}
            for region in self.regions:
                q_hats[region] = torch.zeros((len(self.alphas), cov_window_size + data_size + week_ahead - 1))
                error_seqs[region] = torch.ones((len(self.alphas), cov_window_size + data_size + week_ahead - 1))
                
                # update init error sequence
                if training_params['init_err_seq_w_alpha'] == True:
                    for a in range(len(self.alphas)):
                        error_seqs[region][a] = error_seqs[region][a] * self.alphas[a]
                
                pg_seqs[region] = torch.zeros(cov_window_size + data_size + week_ahead - 1, 2)
                score_seqs[region] = torch.zeros((len(self.alphas), cov_window_size + data_size + week_ahead - 1, 2))
                q_adjs[region] = torch.zeros((len(self.alphas)))
            # eval
            for t in range(data_size):
                if t > training_params['total_steps_limit']:
                    continue
                if use_tta:
                    if use_tta_ffn:
                        q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs = self.tta_forward_pass(
                            data=data,
                            regions2train=self.regions,
                            t=t,
                            cov_window_size=cov_window_size,
                            week_ahead=week_ahead,
                            device=device,
                            model=model,
                            pg_seqs=pg_seqs,
                            error_seqs=error_seqs,
                            score_seqs=score_seqs,
                            q_hats=q_hats,
                            q_adjs=q_adjs,
                            monotonicity_threshold=0.05,
                            monotonicity_score_func=monotonicity_measure_func1,
                            mask=training_params['tta_ffn_mask'],
                            l1_reg=training_params['tta_ffn_l1reg'],
                        )
                    else:
                        q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs = self.tta_forward_pass_no_FNN(
                            data=data,
                            regions2train=self.regions,
                            t=t,
                            cov_window_size=cov_window_size,
                            week_ahead=week_ahead,
                            device=device,
                            model=model,
                            pg_seqs=pg_seqs,
                            error_seqs=error_seqs,
                            score_seqs=score_seqs,
                            q_hats=q_hats,
                            q_adjs=q_adjs,
                            monotonicity_threshold=0.05,
                            monotonicity_score_func=monotonicity_measure_func1,
                        )
                else:
                    q_hats, q_adjs, cur_q_hats, error_seqs, score_seqs, pg_seqs, ys, y_hats, region_idxs = self.forward_pass(
                        data=data,
                        regions2train=self.regions,
                        t=t,
                        cov_window_size=cov_window_size,
                        week_ahead=week_ahead,
                        device=device,
                        model=model,
                        loss_calculator=None,
                        alternate_training=False,
                        epoch=0,
                        pg_seqs=pg_seqs,
                        error_seqs=error_seqs,
                        score_seqs=score_seqs,
                        q_hats=q_hats,
                        q_adjs=q_adjs,
                        eval=True,
                    )
                if t >= data_size - test_weeks:
                    # retrain model if needed
                    if training_params['retrain']:
                        steps_after_val = t - data_size + test_weeks + 1
                        if steps_after_val % training_params['retrain_period'] == 0:
                            if not training_params['retrain_with_saved_weights']:
                                model = self.model.to(device)
                            model = self.retrain_model(training_params, data, model, t-(data_size - test_weeks))
                            model.eval()
                    # forecast (save results)
                    q_hats2save = cur_q_hats.detach().cpu().numpy()
                    y_hats2save = y_hats.detach().cpu().numpy()
                    ys2save = ys.detach().cpu().numpy()
                    for r in range(len(self.regions)):
                        region = self.regions[region_idxs[r]]
                        # denormalize
                        q_hat2save = self.scalar.inv_tranform(q_hats2save[r, :], region, week_ahead, is_score=True)
                        y_hat2save = self.scalar.inv_tranform(y_hats2save[r], region, week_ahead, is_score=False)
                        y2save = self.scalar.inv_tranform(ys2save[r], region, week_ahead, is_score=False)                 
                        all_predictions[region][week_ahead].append((q_hat2save, y_hat2save, y2save))
        return all_predictions


def check_seqs(cur_qhats, Ets, q_hats, scores, region_idxs, regions, target_region, target_alpha_idx):
    for r in range(len(regions)):
        region = regions[region_idxs[r]]
        if region != target_region:
            continue
        print('Print Start')
        print(scores[r, 0])
        print(q_hats[region][target_alpha_idx, :])
        print(cur_qhats[r, target_alpha_idx])
        print(Ets[region][target_alpha_idx, :])
        # q_hats[region][:, idx2update] = cur_q_hats[r, :]
        # prev_seq[region][0, idx2update] = scores[r, 0]
        # for i in range(len(alphas)):
        #     Ets[region][i, idx2update] = int(cur_q_hats[r, i] < scores[r, 0])
        #     prev_seq[region][i+1, idx2update] = float(cur_q_hats[r, i])


def check_inputs(inputs):
    print('input shapes start printing')
    for input in inputs:
        try:
            print(input.shape)
        except:
            print('no shape')
    print('input shapes finish printing')
    