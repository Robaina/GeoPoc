import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean,scatter_add
from torch_geometric.nn import TransformerConv
from data import *

def split_batch(x,batchid):
    x =  x.unsqueeze(0)
    unique_batch_ids = torch.unique(batchid)
    # flag = 0
    batchx = []
    for batch_id in unique_batch_ids:
        batch_indices = torch.nonzero(batchid == batch_id).squeeze()
        # if flag==0:
        #     batchx = x[:,batch_indices]
        #     flag = 1
        # else:
        #     batchx = torch.cat((batchx,x[:,batch_indices]),dim=0)
        batchx.append(x[:,batch_indices])
    return batchx
        
        

class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        # c_V = scatter_add(h_V, batch_id, dim=0)
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)
        # print(h_V.shape)
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention

class GPSoc(nn.Module): # Geometry-aware Protein Sequence optimal condition predictor
    def __init__(self, node_input_dim=2753, edge_input_dim=450, hidden_dim=256, num_layers=2, dropout=0.2, device='cpu',task='temp'):
        super(GPSoc, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)

        self.attention = Attention(hidden_dim,dense_dim=16,n_heads=4)

        self.task = task
        if self.task =="temp":
            self.plus = 101.0
            self.addbase = 4.0
            self.output_dim=1
        else:
            self.output_dim=3
        
        self.add_module("FC_1", nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.add_module("FC_2", nn.Linear(hidden_dim, self.output_dim, bias=True))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, h_V, edge_index, batch_id):
        h_V_geo, h_E = get_geo_feat(X, edge_index)
        h_V = torch.cat([h_V, h_V_geo], dim=-1)
        h_V = self.Graph_encoder(h_V, edge_index, h_E, None, batch_id) 
        
        batchx = split_batch(h_V,batch_id)
        feature_embedding = torch.tensor([]).to(self.device)
        for h_vi in batchx:
            att = self.attention(h_vi)
            h_vi = att @ h_vi 
            h_vi = torch.sum(h_vi,1)
            feature_embedding = torch.cat((feature_embedding,h_vi),dim=0)
        h_V = feature_embedding
        if self.task == "temp":
            h_V = F.leaky_relu(self._modules["FC_1"](h_V))
            output = self._modules["FC_2"](h_V).sigmoid()
            output = output.view([-1])*self.plus+self.addbase
        else:
            h_V = F.elu(self._modules["FC_1"](h_V))
            output = self._modules["FC_2"](h_V)
        
        return output
