import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from forecaster.basicmodels import *
from forecaster.transformer_wrnn_decoder import TransformerSeqEncoder, TransformerEncoderRNNDecoder, MultiViewEncoder
from forecaster.informer import Informer
import copy


class QuantilePredictor(nn.Module):
    def __init__(self, fc_hidden_dim, hidden_dim, output_dim, cumulative_quantiles, actv_type):
        super(QuantilePredictor, self).__init__()
        self.cumulative_quantiles = cumulative_quantiles
        self.fc1 = nn.Linear(hidden_dim,fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc3 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc4 = nn.Linear(fc_hidden_dim, output_dim)
        self.actv = nn.ReLU() if actv_type == 'relu' else nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self, emb):
        """_summary_

        Args:
            emb (_type_): batch size x hidden size
            input_alphas (_type_): batch size x alphas

        Returns:
            _type_: batch size x alphas
        """
        x = self.actv(self.fc1(emb))
        x = self.actv(self.fc2(x))
        x = self.actv(self.fc3(x))
        x = self.relu(self.fc4(x))
        if self.cumulative_quantiles:
            x = torch.flip(x, dims=[1])
            x = torch.cumsum(x, dim=1)
            x = torch.flip(x, dims=[1])
        return x


class TransformerCP(nn.Module):
    def __init__(self, params):
        super(TransformerCP, self).__init__()
        self.with_week_id = params['with_week_id']
        self.alphas = params['alphas']
        self.params = params
        
        # init modules
        if params['x_encoder_type'] == 'transformer':
            self.transformer1 = TransformerEncoderRNNDecoder(params)
        elif params['x_encoder_type'] == 'informer':
            encoder_informer_params = copy.deepcopy(params['encoder_informer'])
            encoder_informer_params = {**encoder_informer_params, **params['informer_shared_params']}
            encoder_informer_params['num_regions'] = params['num_regions']
            encoder_informer_params['num_aheads'] = params['num_aheads']
            encoder_informer_params['with_time_emb'] = params['with_week_id']
            encoder_informer_params['informer_seq_length'] = params['seq_length']
            encoder_informer_params['informer_enc_in'] = params['encoder_input_dim']
            encoder_informer_params['informer_output_dim'] = params['hidden_dim']
            self.transformer1 = Informer(encoder_informer_params, multimodal=True)
        elif params['x_encoder_type'] == 'mvencoder':
            self.transformer1 = MultiViewEncoder(params)
        self.score_encoder = RNNSeqEncoder(
            input_dim=3,
            hidden_dim=params['score_encoder_hidden_dim'],
            num_layers=params['score_encoder_num_layers'],
            output_dim=params['hidden_dim'],
            actv_type=params['actv_type'],
        )
        if params['qhat_encoder_type'] == 'rnn':
            self.qhat_encoder = RNNSeqEncoder(
            input_dim=len(params['alphas']),
            hidden_dim=params['qhat_encoder_hidden_dim'],
            num_layers=params['qhat_encoder_num_layers'],
            output_dim=params['hidden_dim'],
            actv_type=params['actv_type'],
        )
        elif params['qhat_encoder_type'] == 'transformerrnn':
            self.qhat_encoder = TransformerSeqEncoder(
            input_dim=len(params['alphas']),
            hidden_dim=params['qhat_encoder_hidden_dim'],
            num_heads=params['qhat_encoder_num_heads'],
            num_layers=params['qhat_encoder_num_layers'],
            rd_hidden_dim=params['qhat_encoder_rd_hidden_dim'],
            rd_num_layers=params['qhat_encoder_rd_num_layers'],
            rd_steps_ahead=params['num_aheads'],
            rd_output_dim=params['hidden_dim'],
        )
        elif params['qhat_encoder_type'] == 'informer':
            qhat_informer_params = copy.deepcopy(params['qhat_informer'])
            qhat_informer_params = {**qhat_informer_params, **params['informer_shared_params']}
            qhat_informer_params['num_regions'] = params['num_regions']
            qhat_informer_params['num_aheads'] = params['num_aheads']
            qhat_informer_params['with_time_emb'] = params['with_week_id']
            qhat_informer_params['informer_seq_length'] = params['window_size']
            qhat_informer_params['informer_enc_in'] = len(params['alphas'])
            qhat_informer_params['informer_output_dim'] = params['hidden_dim']
            self.qhat_encoder = Informer(
                qhat_informer_params,
                multimodal=False
            )
        self.error_encoder = ErrEncoder(
            input_dim1=len(params['alphas']),
            input_dim2=params['window_size'],
            hidden_dim=params['error_encoder_hidden_dim'],
            output_dim=params['hidden_dim'],
            actv_type=params['actv_type'],
        )
        self.alpha_encoder = FFEncoder(
            input_dim=len(params['alphas']),
            hidden_dim=params['alpha_encoder_hidden_dim'],
            output_dim=params['hidden_dim'],
            actv_type=params['actv_type'],
        )
        self.attn = MultiHeadAttention(
            input_dim=params['hidden_dim'],
            num_heads=8,
        )
        self.attn1 = MultiHeadCrossAttention(
            hidden_dim=params['hidden_dim'],
            num_heads=params['attn1_num_heads']
        )
        self.qmodel = QuantilePredictor(
            fc_hidden_dim=params['q_fc_hidden_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=len(params['alphas']),
            cumulative_quantiles=params['cumulative_quantiles'],
            actv_type=params['actv_type']
        )
        self.tta_model = FFEncoder(
            input_dim=params['hidden_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=len(params['alphas']),
            actv_type='tanh'
        )
        self.deltaQ_encoder = FFEncoder(
            input_dim=len(params['alphas']),
            hidden_dim=params['hidden_dim'],
            output_dim=params['hidden_dim'],
            actv_type='tanh'
        )
    
    def get_embedding(self, x, region_id, week_id, week_ahead_id, err_seq, score_seq, qhat_seq, input_alphas, q_adj):
        use_ainfo = True
        if 'use_ainfo' in self.params:
            use_ainfo = self.params['use_ainfo']
        
        if use_ainfo:
            # encoder_output shape: (batch size x sequence length x hidden size)
            if self.params['x_encoder_type'] == 'informer':
                decoder_output = self.transformer1.forward_multimodal(x, region_id, week_id, week_ahead_id)
            elif self.params['x_encoder_type'] == 'transformer':
                decoder_output = self.transformer1(x, region_id, week_id, week_ahead_id)
            elif self.params['x_encoder_type'] == 'mvencoder':
                decoder_output = self.transformer1(x, region_id, week_id, week_ahead_id)
        deltaq_encoding = self.deltaQ_encoder(q_adj)
        score_encoding = self.score_encoder(score_seq)
        qhat_encoding = self.qhat_encoder(qhat_seq)
        error_encoding = self.error_encoder(err_seq)
        alpha_encoding = self.alpha_encoder(input_alphas)
        
        # data fusion
        if self.params['use_deltaq_emb']:
            emb = torch.stack([score_encoding, qhat_encoding, error_encoding, alpha_encoding, deltaq_encoding], dim=0)
        else:
            emb = torch.stack([score_encoding, qhat_encoding, error_encoding, alpha_encoding], dim=0)
        emb = emb.permute(1, 0, 2)
        emb = self.attn(emb)
        if use_ainfo:
            if self.params['fuse_mv_data']:
                decoder_output = decoder_output[:, None, :]
            emb = self.attn1(emb, decoder_output)
        return emb
    
    def conformalize(self, current_q, err_seq, input_alphas, learning_rate, q_adj):
        """Conformalize the output from model.

        Args:
            current_q (torch.Tensor): model output (batch size x num alphas)
            err_seq (torch.Tensor): previous errors in a window (batch size x window size x num alphas)
            input_alphas (torch.Tensor): (batch size x num alphas)
            learning_rate (float): learning rate for the similar to aci update
            q_adj (torch.Tensor): delta q T (batch size x num alphas)
        """
        avg_err = torch.mean(err_seq, dim=1)
        new_q_adj = q_adj + learning_rate * (avg_err - input_alphas)
        adjusted_q = current_q + new_q_adj
        return adjusted_q, new_q_adj

    def forward(self, x, region_id, week_id, week_ahead_id, err_seq, score_seq, qhat_seq, input_alphas, learning_rate=0, q_adj=0, t=0):
        """_summary_

        Args:
            x (_type_): _description_
            region_id (_type_): _description_
            week_id (_type_): _description_
            week_ahead_id (_type_): _description_
            err_seq (_type_): batch size x window size x num alphas
            score_seq (_type_): batch size x window size x 3 (y, yhat, score)
            qhat_seq (_type_): batch size x window size x num alphas
            input_alphas (_type_): batch size x num alphas
            learning_rate: float
            q_adj: batch size x num alphas

        Returns:
            _type_: _description_
        """
        # use additional information or not. If not, only use scores for forecasting
        emb = self.get_embedding(x, region_id, week_id, week_ahead_id, err_seq, score_seq, qhat_seq, input_alphas, q_adj)
        qs = self.qmodel(emb)
        
        # conformalize the outputs
        if self.params['e2ecf'] == True and t > self.params['window_size'] * 2:
            qs, q_adj = self.conformalize(qs, err_seq, input_alphas, learning_rate=learning_rate, q_adj=q_adj)
        return qs, q_adj
    
    def tta_forward(self, x, region_id, week_id, week_ahead_id, err_seq, score_seq, qhat_seq, input_alphas, learning_rate=0, q_adj=0, t=0):
        """Perform one forward/backward pass of h."""
        # get qs, same as forward
        emb = self.get_embedding(x, region_id, week_id, week_ahead_id, err_seq, score_seq, qhat_seq, input_alphas, q_adj)
        qs = self.qmodel(emb)
        # conformalize the outputs
        if self.params['e2ecf'] == True and t > self.params['window_size'] * 2:
            qs, q_adj = self.conformalize(qs, err_seq, input_alphas, learning_rate=learning_rate, q_adj=q_adj)
        # get h
        hs = self.tta_model(emb)
        return qs, q_adj, hs
        