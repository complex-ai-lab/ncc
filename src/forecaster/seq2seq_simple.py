import torch
import torch.nn as nn
import math


class Seq2Emb(nn.Module):
    """ Sequence to sequence model for forecasting."""

    def __init__(self,
                 x_train_dim,
                 rnn_hidden_dim=32,
                 hidden_dim=32,
                 n_layers=2,
                 bidirectional=True):
        super().__init__()

        self.encoder = EmbedAttenSeq(
            dim_seq_in=x_train_dim,
            rnn_out=rnn_hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """ Forward pass of the model.
            Input:
                x: (batch_size, seq_len, input_size)
                x_mask: (batch_size, seq_len, 1)
                meta: (batch_size, meta_dim)
            Output:
                out: (batch_size, seq_len, out_dim)
        """
        x_embeds = self.encoder.forward(x.transpose(1, 0))
        return x_embeds


class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        """
        Input:
            dim_in: Dimensionality of input sequence
            value_dim: Dimension of value transform
            key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.value_layer.apply(init_weights)
        self.query_layer.apply(init_weights)
        self.key_layer.apply(init_weights)

    def forward(self, seq):
        """
        Input:
            seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        query = self.query_layer(seq_in)
        key = self.key_layer(seq_in)
        value = self.value_layer(seq_in)
        weights = (query @ key.transpose(1, 2)) / math.sqrt(key.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ value).transpose(1, 0)


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module
    """

    def __init__(self,
                 dim_seq_in=5,
                 rnn_out=40,
                 dim_out=50,
                 n_layers=1,
                 bidirectional=False,
                 attn=TransformerAttn,
                 dropout=0):
        """
        Input:
            dim_seq_in: Dimensionality of input sequence 
            dim_metadata: Dimensionality of input metadata 
            dim_out: Dimensionality of output vector
            rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out //
            2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(in_features=self.rnn_out,
                      out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, seqs):
        """
            seqs: Sequence in dimension [Seq len, Batch size, Hidden size]
            metadata: One-hot encoding for region information
        """
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        emb = self.out_layer(latent_seqs)
        return emb