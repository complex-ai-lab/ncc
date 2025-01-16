import torch
import torch.nn as nn
import dgl
import math


class FFEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, actv_type='relu'):
        super(FFEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU() if actv_type == 'relu' else nn.Tanh()
    
    def forward(self, Ets):
        """

        Args:
            Ets (_type_): batch size x coverage window size

        Returns:
            _type_: batch size x hidden size
        """
        x = self.activation(self.fc1(Ets))
        x = self.fc2(x)
        return x


class ErrEncoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim, actv_type='relu'):
        super(ErrEncoder, self).__init__()
        self.shape1 = input_dim2 * hidden_dim
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(input_dim2*hidden_dim, output_dim)
        self.actv = nn.ReLU() if actv_type == 'relu' else nn.Tanh()
    
    def forward(self, Ets):
        """

        Args:
            err_seq (_type_): batch size x coverage window size x alphas

        Returns:
            _type_: batch size x hidden size
        """
        x = self.actv(self.fc1(Ets))
        x = self.actv(self.fc2(x)).reshape(-1, self.shape1)
        x = self.actv(self.fc3(x))
        return x



class RNNSeqEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, actv_type='relu'):
        super(RNNSeqEncoder, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(num_layers, 1)
        self.actv = nn.ReLU() if actv_type == 'relu' else nn.Tanh()

    def forward(self, x):
        """

        Args:
            x (_type_): batch size x sequence length x input dim

        Returns:
            _type_: batch size x hidden dim
        """
        x = self.rnn(x)[-1]
        x = x.permute(1, 2, 0)
        x = self.actv(self.fc3(x))[:, :, 0]
        x = self.actv(self.fc1(x))
        x = self.fc2(x)
        return x


class EmbGCNEncoder(nn.Module):
    """
    Encoder for categorical values with adj graph
    """

    def __init__(
        self,
        in_size: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation='relu',
    ):
        super(EmbGCNEncoder, self).__init__()
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.emb_layer = nn.Embedding(in_size, hidden_dim)
        self.gcn_layers = [dgl.nn.GraphConv(hidden_dim, out_dim)]
        self.drop_layer = nn.Dropout(p=dropout)
        for _ in range(num_layers - 1):
            self.gcn_layers.append(dgl.nn.GraphConv(out_dim, out_dim))

    def forward(self, batch, graph):
        embs = self.emb_layer(batch)
        for i in range(len(self.gcn_layers)):
            embs = self.activation(
                self.drop_layer(self.gcn_layers[i](graph, embs))
            )
        return embs


class SelfAttention(nn.Module): 
    def __init__(self, input_dim): 
        super(SelfAttention, self).__init__() 
        self.input_dim = input_dim 
        self.query = nn.Linear(input_dim, input_dim) 
        self.key = nn.Linear(input_dim, input_dim) 
        self.value = nn.Linear(input_dim, input_dim) 
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): batch size x sequence length x input dim

        Returns:
            _type_: same as input
        """
        queries = self.query(x) 
        keys = self.key(x) 
        values = self.value(x) 
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5) 
        attention = self.softmax(scores) 
        weighted = torch.bmm(attention, values)
        weighted = torch.sum(weighted, dim=1)
        return weighted


class CrossAtten4AuxEmb(nn.Module):
    """
    Attention used to fuse two embeddings
    """

    def __init__(self, input_dim) -> None:
        super(CrossAtten4AuxEmb, self).__init__()
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

    def forward(self, main_emb, aux_emb):
        """Forward pass

        Args:
            main_emb (torch.Tensor): the main embedding to which the auxiliary embeddings will be aligned to. (batch size x hidden size)
            aux_emb (torch.Tensor): a list of auxiliary embeddings from multi-view data. (batch size x # views x hidden size)
        """
        main_emb = main_emb[:, None, :]
        all_emb = torch.concat([main_emb, aux_emb], dim=1)
        querys = self.query_layer(main_emb)
        keys = self.key_layer(all_emb)
        values = self.value_layer(all_emb)
        atten = torch.bmm(querys, keys.transpose(1, 2))
        atten = torch.softmax(atten, 2)
        fused_emb = torch.bmm(atten, values)[:, 0, :]
        return fused_emb


class LatentAtten(nn.Module):
    """
    Attention on latent representation
    """

    def __init__(self, input_dim) -> None:
        super(LatentAtten, self).__init__()
        self.input_dim = input_dim
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.query_layer = nn.Linear(input_dim, input_dim)

    def forward(self, h_M, h_R):
        key = self.key_layer(h_M)
        query = self.query_layer(h_R)
        atten = (key @ query.transpose(0, 1)) / math.sqrt(self.input_dim)
        atten = torch.softmax(atten, 1)
        print(atten.shape)
        weighted = torch.bmm(atten, h_M)
        weighted = torch.sum(weighted, dim=1)
        return weighted


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(input_dim=input_dim) 
             for _ in range(num_heads)]
        )
        self.fc = nn.Linear(num_heads, 1)

    def forward(self, x):
        embs = torch.stack([head(x) for head in self.heads], dim=-1)
        return self.fc(embs)[:, :, 0]


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadCrossAttention, self).__init__()
        self.heads = nn.ModuleList(
            [CrossAtten4AuxEmb(input_dim=hidden_dim)
            for _ in range(num_heads)]
        )
        self.fc = nn.Linear(num_heads, 1)

    def forward(self, x, aux):
        # x: (B x H), aux: (B x C x H)
        embs = torch.stack([head(x, aux) for head in self.heads], dim=-1)
        # print(embs.shape)
        return self.fc(embs)[:, :, 0]