import torch
import torch.nn as nn
import numpy as np


class LossTracker:
    def __init__(self, loss_types=['qloss', 'closs', 'eloss', 'mloss']):
        self.loss_types = loss_types
        self.additional_loss_type = 'total_loss'
        self.train_loss = {}
        self.val_loss = {}
        self.tmp_train_loss = {}
        self.tmp_val_loss = {}
        for loss_type in loss_types:
            self.train_loss[loss_type] = []
            self.val_loss[loss_type] = []
        self.train_loss[self.additional_loss_type] = []
        self.val_loss[self.additional_loss_type] = []

    def add_train_loss(self, losses_in_current_epoch, convert2np=True):
        qloss, closs, eloss, mloss = losses_in_current_epoch
        if convert2np:
            qloss = qloss.detach().cpu().numpy()
            closs = closs.detach().cpu().numpy()
            eloss = eloss.detach().cpu().numpy()
            mloss = mloss.detach().cpu().numpy()
            closs = np.nan if closs > 1e7 else closs
        self.tmp_train_loss['qloss'].append(qloss)
        self.tmp_train_loss['closs'].append(closs)
        self.tmp_train_loss['eloss'].append(eloss)
        self.tmp_train_loss['mloss'].append(mloss)
        
    def add_val_loss(self, losses_in_current_epoch, convert2np=True):
        qloss, closs, eloss, mloss = losses_in_current_epoch
        if convert2np:
            qloss = qloss.detach().cpu().numpy()
            closs = closs.detach().cpu().numpy()
            eloss = eloss.detach().cpu().numpy()
            mloss = mloss.detach().cpu().numpy()
            closs = np.nan if closs > 1e7 else closs
        self.tmp_val_loss['qloss'].append(qloss)
        self.tmp_val_loss['closs'].append(closs)
        self.tmp_val_loss['eloss'].append(eloss)
        self.tmp_val_loss['mloss'].append(mloss)
    
    def epoch_start(self):
        for loss_type in self.loss_types:
            self.tmp_train_loss[loss_type] = []
            self.tmp_val_loss[loss_type] = []
    
    def epoch_end(self, verbose=False):
        total_train_loss = 0
        total_val_loss = 0
        for loss_type in self.loss_types:
            tmp_train_loss = np.nanmean(self.tmp_train_loss[loss_type])
            tmp_val_loss = np.nanmean(self.tmp_val_loss[loss_type])
            self.train_loss[loss_type].append(tmp_train_loss)
            self.val_loss[loss_type].append(tmp_val_loss)
            total_train_loss += tmp_train_loss
            total_val_loss += tmp_val_loss
        self.train_loss[self.additional_loss_type].append(total_train_loss)
        self.val_loss[self.additional_loss_type].append(total_val_loss)
        if verbose:
            print(f'Training total loss: {total_train_loss}, cov loss: {np.mean(self.tmp_train_loss["closs"])}, quantile loss: {np.mean(self.tmp_train_loss["qloss"])}')
        return total_val_loss
    
    def get_losses(self):
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }


# losses
class QuantileLoss(nn.Module):
    def __init__(self, alphas):
        super().__init__()
        self.quantiles = []
        for alpha in alphas:
            self.quantiles.append(1-alpha)
        
    def forward(self, preds, target):
        # assume preds in the shape of (batch size, quantiles size), target in the shape of (batch size)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.size(1) == len(self.quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:] - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class CoverageLoss(nn.Module):
    def __init__(self, K, seq_length, alphas, device):
        """
        K is used in sigmoid function.
        seq_length is W + 1.
        alphas: (A)
        """
        super().__init__()
        self.K = K
        self.seq_length = seq_length
        self.alphas = torch.Tensor(alphas).to(device)
    
    def forward(self, cur_Ets, q_hat, q):
        e_prev = torch.mean(cur_Ets, dim=1)
        e_hat = torch.sigmoid((q - q_hat) / self.K)
        # clamp so that e_hat is within a valid range
        e_hat = torch.clamp(e_hat, min=0.01, max=0.99)
        tmp = self.alphas[None, :] - e_prev
        e = torch.where(tmp > 0, 1, 0)
        # f = torch.where(torch.abs(tmp)>=self.thre, 1, 0)
        closs = -1 * torch.nanmean(e*torch.log(e_hat) + (1-e)*torch.log(1-e_hat))
        # print('Print Start')
        # print(f'closs in func: {closs}')
        # print(e_hat[0])
        # print(torch.log(e_hat)[0])
        # print(torch.log(1-e_hat)[0])
        # if torch.isnan(closs).sum() > 0 or closs > 1e5:
        #     print('Invalid closs detected in loss tracker')
        #     np_ehat = e_hat.detach().cpu().numpy()
        #     print(np.min(np_ehat))
        #     print(np.max(np_ehat))
        #     import matplotlib.pyplot as plt
        #     # plt.hist(np_ehat, bins=200)
        #     # plt.show()
        #     print('end printing')
        return closs
        
        
    # def forward(self, cur_Ets, q_hat, q):
    #     """
    #     Calculate coverage loss.
    #     Denote the number of alphas by A, the number of batches by B, the window size (sequence length of errors) is W.
    #     cur_Ets: errors in a fixed window. (B x W x A)
    #     q_hat: prediction intervals. (B x A)
    #     q: uncertainty scores, |y - y_hat| by default. (B x 1)
    #     """
    #     # calculate previous errors
    #     prev_errors = torch.sum(cur_Ets, dim=1)
    #     # print(prev_errors)
        
    #     # current error estimated by sigmoid function
    #     cur_error = torch.sigmoid((q - q_hat) / self.K)
        
    #     # mean squared loss
    #     avg_error = (prev_errors + cur_error) / self.seq_length
    #     return torch.mean((avg_error - self.alphas)**2)


class EfficiencyLoss(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        
    def forward(self, q_hat, q):
        """
        Calculate efficiency loss.
        """
        # current error estimated by sigmoid function
        cur_error = torch.sigmoid((q - q_hat) / self.K)
        
        # efficiency loss
        eloss = torch.square(q_hat - q) * (1 - cur_error)
        return torch.mean(eloss)


class MonotonicityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q_hat, input_alphas):
        """
        # TODO: assumes batch size = 1
        Calculate monotonicity loss.
        Denote the number of alphas by A, the number of batches by B.
        q_hat: prediction intervals. (B x A)
        input_alphas: alphas. (B x A)
        """
        # get grad from torch
        # q_hat.backward(torch.ones(q_hat.shape).to(q_hat.device), retain_graph=True)
        # alpha_grad = input_alphas.grad
        # print(alpha_grad)
        
        # use delta alpha to approach deviation
        q_diff = torch.diff(q_hat, dim=1)
        a_diff = torch.diff(input_alphas, dim=1)
        dev = q_diff / a_diff
        mloss = nn.functional.relu(dev)
        return torch.mean(mloss)


class LossCalculator:
    def __init__(self, training_params, alphas):
        device = torch.device(training_params['device'])
        # loss functions
        if 'eK' not in training_params:
            training_params['eK'] = training_params['K']
        self.q_loss_fn = QuantileLoss(alphas)
        self.c_loss_fn = CoverageLoss(training_params['K'], training_params['cov_window_size']+1, alphas, device)
        self.e_loss_fn = EfficiencyLoss(training_params['eK'])
        self.m_loss_fn = MonotonicityLoss()
        # loss factors
        self.q_loss_factor = training_params['qloss_factor']
        self.c_loss_factor = training_params['closs_factor']
        self.e_loss_factor = training_params['eloss_factor']
        self.m_loss_factor = training_params['mloss_factor']
    
    def cal_loss(self, cur_q_hats, cur_Ets, input_alphas, score, return_all_losses=True, alter_training=False, epoch=0):
        opt_qloss = 1
        opt_closs = 1
        opt_eloss = 1
        if alter_training:
            opt_qloss = 0
            opt_closs = 0
            opt_eloss = 0
            if epoch % 3 == 0:
                opt_qloss = 1
            elif epoch % 3 == 1:
                opt_closs = 1
            elif epoch % 3 == 2:
                opt_eloss = 1
        qloss = opt_qloss * self.q_loss_factor * self.q_loss_fn(cur_q_hats, score[:, 0])
        closs = opt_closs * self.c_loss_factor * self.c_loss_fn(cur_Ets, cur_q_hats, score)
        eloss = opt_eloss * self.e_loss_factor * self.e_loss_fn(cur_q_hats, score)
        mloss = self.m_loss_factor * self.m_loss_fn(cur_q_hats, input_alphas)
        loss = qloss + closs + eloss + mloss
        if torch.sum(torch.isnan(loss)) > 0:
            print(f'Loss is nan')
            print(f'qloss: {qloss}, closs: {closs}, eloss: {eloss}, mloss: {mloss}')
        # loss = qloss + eloss + mloss
        if return_all_losses:
            return loss, qloss, closs, eloss, mloss
            # return loss, qloss, mloss, eloss, mloss
        return loss