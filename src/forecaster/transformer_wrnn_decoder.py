import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from forecaster.basicmodels import RNNSeqEncoder, MultiHeadAttention
from forecaster.seq2seq_simple import Seq2Emb

class PositionalEncoding(nn.Module):
    """
    From: https://anonymous.4open.science/r/EmbedTS-3F5D/src/embedts/models/transformers/layers.py
    
    Positional encoding as described in "Attention is all you need"
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # print(self.pe.shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TimeAwareEmbedding(nn.Module):

    def __init__(self, hidden_dim, num_weeks):
        super(TimeAwareEmbedding, self).__init__()
        self.week_embedding = nn.Embedding(num_weeks + 1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, week_ids):
        week_embedded = self.week_embedding(week_ids)
        return self.fc(week_embedded)


class TemporalConvolution(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(TemporalConvolution, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Permute the input for 1D convolution
        x = x.permute(1, 2, 0)
        conv_out = self.conv1d(x)
        # Revert the permutation
        conv_out = conv_out.permute(2, 0, 1)
        return conv_out


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_regions, num_aheads=4, with_time_emb=True):
        super(TransformerEncoder, self).__init__()
        self.with_time_emb = with_time_emb
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.region_embedding = nn.Embedding(num_regions, hidden_dim)
        self.time_embedding = TimeAwareEmbedding(hidden_dim, num_weeks=53)
        self.week_ahead_embedding = TimeAwareEmbedding(hidden_dim, num_weeks=num_aheads)
        self.temporal_conv = TemporalConvolution(hidden_dim, hidden_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

    def forward(self, x, region_ids, week_ids, week_ahead_id):
        # print(type(region_ids))
        x = self.embedding(x)
        x = self.temporal_conv(x)  # temporal convolution
        rs_embedded = self.region_embedding(region_ids)
        if self.with_time_emb:
            time_embedded = self.time_embedding(week_ids)
            x = x + time_embedded
        week_ahead_embedded = self.week_ahead_embedding(week_ahead_id)
        x = x + rs_embedded + week_ahead_embedded   # Add the embeddings to the input
        
        # positional encoding
        x = x.permute(1, 0, 2)
        # print(x.shape)
        x = self.pos_encoder(x)
        
        # transformer encoder
        # encoder_output = self.encoder(x, src_key_padding_mask=x_mask)
        encoder_output = self.encoder(x)
        return encoder_output.permute(1, 0, 2)


class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, steps_ahead, output_dim):
        super(RNNDecoder, self).__init__()
        self.steps_ahead = steps_ahead
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(steps_ahead, output_dim)

    def forward(self, encoder_output):
        decoder_output = self.rnn(encoder_output)[0]
        decoder_output = decoder_output[:, -self.steps_ahead:, :]
        decoder_output = self.fc1(decoder_output)
        decoder_output = decoder_output[:, :, 0]
        decoder_output = self.fc2(decoder_output)
        return decoder_output


class TransformerSeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, rd_hidden_dim, rd_num_layers, rd_steps_ahead, rd_output_dim) -> None:
        """
        Use a transformer model to encode qhats. This module includes a transformer encoder and a rnn decoder.
        """
        super(TransformerSeqEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.temporal_conv = TemporalConvolution(hidden_dim, hidden_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.rnn_decoder = RNNDecoder(
            input_dim=hidden_dim,
            hidden_dim=rd_hidden_dim,
            num_layers=rd_num_layers,
            steps_ahead=rd_steps_ahead,
            output_dim=rd_output_dim,
        )
    
    def forward(self, x):
        """Encode qhats

        Args:
            x (_type_): qhats in the shape of (batch size x window size x num alphas)
        """
        x = self.embedding(x)
        x = self.temporal_conv(x)  # temporal convolution
        
        # positional encoding
        x = x.permute(1, 0, 2)
        # print(x.shape)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.rnn_decoder(x)
        return x


class TransformerEncoderRNNDecoder(nn.Module):
    def __init__(self, params) -> None:
        super(TransformerEncoderRNNDecoder, self).__init__()
        self.encoder = TransformerEncoder(
            input_dim=params['encoder_input_dim'],
            hidden_dim=params['encoder_hidden_dim'],
            num_layers=params['encoder_num_layers'],
            num_heads=params['encoder_heads'],
            num_regions=params['num_regions'],
            num_aheads=params['num_aheads'],
            with_time_emb=params['with_week_id'],
        )
        self.decoder = RNNDecoder(
            input_dim=params['encoder_hidden_dim'],
            hidden_dim=params['decoder_hidden_dim'],
            num_layers=params['decoder_num_layers'],
            steps_ahead=params['num_aheads'],
            output_dim=params['hidden_dim'],
        )
    
    def forward(self, x, region_id, week_id, week_ahead_id):
        encoder_output = self.encoder(x, region_id, week_id, week_ahead_id)
        # encoder_output shape: (batch size x sequence length x hidden size)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
    
    
    def forward_multimodal(self, x, region_id, week_id, week_ahead_id):
        encoder_output = self.encoder(x, region_id, week_id, week_ahead_id)
        # encoder_output shape: (batch size x sequence length x hidden size)
        decoder_output = self.decoder(encoder_output)
        return decoder_output



################
##New Encoders##
################

class MultiViewEncoder(nn.Module):
    def __init__(self, params) -> None:
        super(MultiViewEncoder, self).__init__()
        self.with_time_emb=params['with_week_id']
        self.seq_encoder = None
        self.fuse_mv_data = params['fuse_mv_data']
        if params['x_encoder_seq_encoder_type'] == 'seq2seq':
            self.seq_encoder = Seq2Emb(
                x_train_dim=params['encoder_input_dim'],
                rnn_hidden_dim=params['encoder_hidden_dim'],
                hidden_dim=params['hidden_dim'],
                n_layers=params['encoder_num_layers'],
            )
        elif params['x_encoder_seq_encoder_type'] == 'gru':
            self.seq_encoder = RNNSeqEncoder(
                input_dim=params['encoder_input_dim'],
                hidden_dim=params['encoder_hidden_dim'],
                num_layers=params['encoder_num_layers'],
                output_dim=params['hidden_dim'],
            )
        self.region_embedding = nn.Embedding(params['num_regions'], params['hidden_dim'])
        self.time_embedding = TimeAwareEmbedding(params['hidden_dim'], num_weeks=53)
        self.week_ahead_embedding = TimeAwareEmbedding(params['hidden_dim'], num_weeks=params['num_aheads'])
        self.multiheadattn = MultiHeadAttention(params['hidden_dim'], num_heads=16)
    
    def forward(self, x, region_id, week_id, week_ahead_id):
        seq_embedded = self.seq_encoder(x)
        rs_embedded = self.region_embedding(region_id)
        week_ahead_embedded = self.week_ahead_embedding(week_ahead_id)
        if self.with_time_emb:
            time_embedded = self.time_embedding(week_id)
            emb = torch.stack([seq_embedded, rs_embedded[:, 0, :], week_ahead_embedded[:, 0, :], time_embedded[:, 0, :]], dim=0)
        else:
            emb = torch.stack([seq_embedded, rs_embedded[:, 0, :], week_ahead_embedded[:, 0, :]], dim=0)
        emb = emb.permute(1, 0, 2) # (batch size x num views x hidden dim)
        if self.fuse_mv_data:
            emb = self.multiheadattn(emb)
        return emb