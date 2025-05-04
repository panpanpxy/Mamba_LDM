from .models.gcl_mamba import E_GCL_mamba, unsorted_segment_sum
import torch
from torch import nn
import random

class E_GCL_mask(E_GCL_mamba):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL_mamba.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, bi=False,order_method='No',d_state=64, dropout=0, mamba_mlp=False,act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        src, dst = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, src, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        src, dst = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        mi,edge_feat = self.edge_model(h,edge_index, radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask
        if self.order_method == 'degree':
            # print('-----------------DEGREE!!!!!!!---------------------')
            degrees = torch.zeros(len(h), dtype=torch.int32).cuda()
            degrees.scatter_add_(0, dst, torch.ones(dst.size(0), dtype=torch.int32).cuda())
            degrees.scatter_add_(0, src, torch.ones(src.size(0), dtype=torch.int32).cuda())
            sorted_indices = torch.argsort(degrees)
            sorted_h = h[sorted_indices]
            h = sorted_h
            sorted_mi = mi[sorted_indices]
            mi = sorted_mi

        elif self.order_method == 'degree_with_shuffle':
            # print('-------------DEGREE with SHUFFLE!!!!!!!----------------')
            degrees = torch.zeros(len(h), dtype=torch.int32).cuda()
            degrees.scatter_add_(0, dst, torch.ones(dst.size(0), dtype=torch.int32).cuda())
            degrees.scatter_add_(0, src, torch.ones(src.size(0), dtype=torch.int32).cuda())
            sorted_indices = torch.argsort(degrees)
            unique_degrees = degrees[sorted_indices].unique(sorted=True)
            new_indices = []
            for degree in unique_degrees:
                same_degree_indices = sorted_indices[degrees[sorted_indices] == degree]
                same_degree_indices_list = same_degree_indices.tolist()
                random.shuffle(same_degree_indices_list)
                new_indices.extend(same_degree_indices_list)
            new_indices = torch.tensor(new_indices).cuda()
            sorted_h = h[new_indices]
            h = sorted_h
            sorted_mi = mi[sorted_indices]
            mi = sorted_mi

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        #h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_attr is not None:
            h = self.node_mlp(torch.cat([mi, h, node_attr], dim=-1))
        else:
            h = self.node_mlp(torch.cat([mi, h], dim=-1))
        h = self.pre_norm(h)
        h = torch.clamp(h, min=-10, max=10)
        if self.bi:
            output2=self.mamba2(h.unsqueeze(0)).squeeze(0)
            output3=self.mamba2(torch.flip(h,[0]).unsqueeze(0)).squeeze(0)
            output=h+output2+output3
        else:
            output=self.mamba2(h.unsqueeze(0)).squeeze(0)
        if self.dropout:
            output=self.out_dropout(output)
        if self.mamba_mlp:
            output=self.mamba_merge_mlp(torch.cat([h,output],-1))

        return output, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_mamba_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_mamba_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_mamba_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class EGNN_mamba(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_mamba, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_mamba_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_mamba_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_mamba_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class Naive(nn.Module):
    def __init__(self, device):
        super(Naive, self).__init__()
        self.device = device
        self.linear = nn.Linear(1, 1)
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        node_mask = node_mask.view(-1, n_nodes)
        bs, n_nodes = node_mask.size()
        x = torch.zeros(bs, 1).to(self.device)
        return self.linear(x).squeeze(1)


class NumNodes(nn.Module):
    def __init__(self, device, nf=128):
        super(NumNodes, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(1, nf)
        self.linear2 = nn.Linear(nf, 1)
        self.act_fn = nn.SiLU()
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        reshaped_mask = node_mask.view(-1, n_nodes)
        nodesxmol = torch.sum(reshaped_mask, dim=1).unsqueeze(1)/29
        x = self.act_fn(self.linear1(nodesxmol))
        return self.linear2(x).squeeze(1)