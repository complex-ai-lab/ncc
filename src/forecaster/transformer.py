import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,
                 num_regions):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.region_embedding = nn.Embedding(num_regions, hidden_dim)
        self.time_embedding = TimeAwareEmbedding(hidden_dim, num_weeks=53)
        self.temporal_conv = TemporalConvolution(hidden_dim, hidden_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

    def forward(self, x, x_mask, region_ids, week_ids):
        region_embedded = self.region_embedding(region_ids)
        rs_embedded = region_embedded.unsqueeze(1)
        time_embedded = self.time_embedding(week_ids)
        time_embedded = time_embedded.unsqueeze(1)
        
        x = self.embedding(x)
        x = self.temporal_conv(x)  # temporal convolution
        x = x + time_embedded + rs_embedded   # Add the embeddings to the input
        
        # positional encoding
        x = x.permute(1, 0, 2)
        # print(x.shape)
        x = self.pos_encoder(x)
        
        # transformer encoder
        x_mask = x_mask[:, :, 0]
        # encoder_output = self.encoder(x, src_key_padding_mask=x_mask)
        encoder_output = self.encoder(x, src_key_padding_mask=x_mask)
        return encoder_output.permute(1, 0, 2)


class FCDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, input_seq_length):
        super(FCDecoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.fc3 = nn.Linear(input_seq_length*int(hidden_dim/4), output_dim)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        x = encoder_output
        # residual connection
        decoder_output = self.fc2(self.relu(self.fc(x)) + encoder_output)
        decoder_output = decoder_output.reshape(decoder_output.shape[0], -1)
        decoder_output = self.fc3(decoder_output)
        return decoder_output


class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNDecoder, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_output):
        decoder_output = self.rnn(encoder_output)[0]
        # print(decoder_output.shape)
        decoder_output = decoder_output[:, -self.output_dim:, :]
        decoder_output = self.fc(decoder_output)
        decoder_output = decoder_output[:, :, 0]
        return decoder_output


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, output_dim, num_regions):
        super(TransformerDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.region_embedding = nn.Embedding(num_regions, hidden_dim)
        self.time_embedding = TimeAwareEmbedding(hidden_dim, num_weeks=53)
        
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, dropout=0.3)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, memory, region_ids, week_ids):
        
        region_embedded = self.region_embedding(region_ids)
        rs_embedded = region_embedded.unsqueeze(1)
        time_embedded = self.time_embedding(week_ids)
        time_embedded = time_embedded.unsqueeze(1)
        
        x = torch.zeros((memory.shape[0], memory.shape[1], self.hidden_dim)).to(memory.device)
        x = x + time_embedded + rs_embedded   # Add the embeddings to the input
        
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        
        memory = memory.permute(1, 0, 2)
        decoder_output = self.decoder(x, memory)

        decoder_output = decoder_output.permute(1, 0, 2)[:, -self.output_dim:, :]
        decoder_output = self.fc(decoder_output)[:, :, 0]
        return decoder_output


class TransformerEncoderDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, seq_length, num_layers, num_heads, num_regions, dr_hidden_dim, dr_layers, dt_layers, decoder_name='transformer'):
        super(TransformerEncoderDecoder, self).__init__()
        
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads, num_regions)
        
        self.decoder_name = decoder_name
        # decoder input shape: (batch size x sequence length x hidden size)
        if decoder_name == 'fc':
            self.decoder = FCDecoder(output_dim, hidden_dim, seq_length)
        if decoder_name == 'rnn':
            self.decoder = RNNDecoder(
                input_dim=hidden_dim,
                hidden_dim=dr_hidden_dim,
                num_layers=dr_layers,
                output_dim=output_dim,
            )
        if decoder_name == 'transformer':
            self.decoder = TransformerDecoder(
                input_dim=hidden_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=dt_layers,
                output_dim=output_dim,
                num_regions=num_regions,
            )

    def forward(self, x, x_mask, region_ids, week_ids):
        encoder_output = self.encoder(x, x_mask, region_ids, week_ids)
        if self.decoder_name == 'transformer':
            decoder_output = self.decoder(encoder_output, region_ids, week_ids)
        else:
            decoder_output = self.decoder(encoder_output)
        return decoder_output