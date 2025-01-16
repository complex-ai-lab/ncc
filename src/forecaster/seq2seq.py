import torch
import torch.nn as nn
import math


class Seq2seq(nn.Module):
    """ Sequence to sequence model for forecasting."""

    def __init__(self,
                 metas_train_dim,
                 x_train_dim,
                 device,
                 weeks_ahead=4,
                 hidden_dim=32,
                 out_layer_dim=32,
                 out_dim=1,
                 n_layers=2,
                 bidirectional=True):
        super().__init__()

        self.device = device

        self.weeks_ahead = weeks_ahead

        self.encoder = EmbedAttenSeq(
            dim_seq_in=x_train_dim,
            dim_metadata=metas_train_dim,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=2 * hidden_dim,  # divides by 2 if bidirectional
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )

        out_layer_width = out_layer_dim
        self.out_layer = [
            nn.Linear(in_features=out_layer_width,
                      out_features=out_layer_width // 2),
            nn.ReLU(),
            nn.Linear(in_features=out_layer_width // 2, out_features=out_dim),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)

    def forward(self, x, x_mask, meta, output_emb=False):
        """ Forward pass of the model.
            Input:
                x: (batch_size, seq_len, input_size)
                x_mask: (batch_size, seq_len, 1)
                meta: (batch_size, meta_dim)
            Output:
                out: (batch_size, seq_len, out_dim)
        """
        x_embeds = self.encoder.forward_mask(x.transpose(1, 0), x_mask.transpose(1, 0), meta)
        # length of sequence is self.weeks_ahead (4 weeks)
        position = torch.arange(1, self.weeks_ahead + 1).repeat(x_embeds.shape[0], 1).unsqueeze(2)
        # create a position tensor of shape (batch_size, seq_len, 1) with values 0 to 1
        position = ((position - position.min()) / (position.max() - position.min())).to(self.device)
        emb = self.decoder(position, x_embeds)
        out = self.out_layer(emb)
        if output_emb:
            return out, emb
        return out


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

    def forward_mask(self, seq, mask):
        """
        Follows https://medium.com/mlearning-ai/how-do-self-attention-masks-work-72ed9382510f
        
        Input:
            seq: Sequence in dimension [Seq len, Batch, Hidden size]
            mask: Mask in dimension [Batch, Seq len, 1]
        """
        seq_in = seq.transpose(0, 1)
        query = self.query_layer(seq_in)
        key = self.key_layer(seq_in)
        value = self.value_layer(seq_in)
        weights = (query @ key.transpose(1, 2)) / math.sqrt(key.shape[-1])
        # add mask to weights -- mask has infinity values
        mask = mask.transpose(0, 1)
        mask = mask.repeat(1, 1, weights.shape[2])
        weights = weights + mask.transpose(2, 1)
        # softmax with mask
        weights = torch.softmax(weights, -1)
        return (weights @ value).transpose(1, 0)


class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module
    """

    def __init__(self,
                 dim_seq_in=5,
                 dim_metadata=50,
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
        self.dim_metadata = dim_metadata
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
            nn.Linear(in_features=self.rnn_out + self.dim_metadata,
                      out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, seqs, metadata):
        """
            seqs: Sequence in dimension [Seq len, Batch size, Hidden size]
            metadata: One-hot encoding for region information
        """
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward_mask(self, seqs, mask, metadata):
        """
        Input:
            seqs: Sequence in dimension [Seq len, Batch size, Hidden size]
            mask: Sequence in dimesnion [Seq len, Batch size, Hidden size]
                Mask to unused time series
            metadata: One-hot encoding for region information

        Returns:
            out: Output of the model
            latent_seqs: Used to initialize decoder
        """
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out


class DecodeSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """

    def __init__(self,
                 dim_seq_in=5,
                 dim_metadata=50,
                 rnn_out=40,
                 dim_out=5,
                 n_layers=1,
                 bidirectional=False,
                 dropout=0.0):
        """
        Input:
            dim_seq_in: Dimensionality of input vector
            dim_out: Dimensionality of output vector
            dim_metadata: Dimensions of metadata for all sequences
            rnn_out: output dimension for rnn
        """
        super(DecodeSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.act_fcn = nn.Tanh()

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out //
            2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = [
            nn.Linear(in_features=self.rnn_out, out_features=self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)

    def forward(self, position, hidden):
        """
        Input:
            position: Sequence in dimension [Batch size, Seq len, 1]
            hidden: Hidden state from encoder [Batch size, Hidden size]
        Returns:
            latent_seqs: Output of the model
        """
        inputs = position.transpose(1, 0)
        if self.bidirectional:
            h0 = hidden.expand(2, -1, -1).contiguous()
        else:
            h0 = hidden.unsqueeze(0)
        # Take last output from GRU
        latent_seqs = self.rnn(inputs, h0)[0]
        # print(latent_seqs.shape)
        latent_seqs = latent_seqs.transpose(1, 0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs
