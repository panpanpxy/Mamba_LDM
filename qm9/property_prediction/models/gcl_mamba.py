import random

from torch import nn
import torch
from mamba_ssm import Mamba
from torch_scatter import scatter_sum


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out


class E_GCL_mamba(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, bi=False,order_method='No',d_state=64, dropout=0, mamba_mlp=False,act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL_mamba, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        self.bi=bi
        self.d_state=d_state
        self.dropout=dropout
        self.mamba_mlp=mamba_mlp
        self.order_method=order_method


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf*2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.mamba_mlp:
            self.mamba_merge_mlp=nn.Sequential(
                nn.Linear(hidden_nf*2, hidden_nf),
            )

        self.mamba2=Mamba(d_model=hidden_nf,
                          d_state=64,
                          d_conv=4,
                          expand=1,
                          )
        self.mamba_norm = nn.LayerNorm(hidden_nf)
        self.pre_norm = nn.LayerNorm(hidden_nf)

        if self.dropout !=0:
            self.out_dropout=nn.Dropout(p=self.dropout)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, x, edge_index, radial,edge_attr):
        dst, src = edge_index
        hi, hj = x[dst], x[src]
        if edge_attr is None:  # Unused.
            out = torch.cat([hj, hi,radial], dim=1)
        else:
            out = torch.cat([hj, hi, radial,edge_attr], dim=1)
        # 更新后的边特征
        mij = self.edge_mlp(out)  # mij mi dim=hidden_nf
            # 边权重的存在概率(取值0~1）,使用attention 相当于加入边的权重 att_val作为加权因子，确保更重要的边对目标节点特征的贡献更大
        if self.attention:
            att_val = self.att_mlp(mij)
            edge_feat=mij*att_val
            mi = scatter_sum(edge_feat, dst, dim=0, dim_size=x.shape[0])
        else:
            mi = scatter_sum(mij, dst, dim=0, dim_size=x.shape[0])
        return mi, edge_feat

    #def node_model(self, x, edge_index, edge_attr, node_attr):
    #    row, col = edge_index
    #    agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
    #    if node_attr is not None:
    #        agg = torch.cat([x, agg, node_attr], dim=1)
    #    else:
    #        agg = torch.cat([x, agg], dim=1)
    #    out = self.node_mlp(agg)
    #    if self.recurrent:
    #        out = x + out
    #    return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        src, dst = edge_index
        hi,hj=h[src],h[dst]
        radial, coord_diff = self.coord2radial(edge_index, coord)

        mi,edge_feat= self.edge_model(h, edge_index, radial, edge_attr)

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

        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        if node_attr is not None:
            h = self.node_mlp(torch.cat([mi, h, node_attr], dim=-1))
        else:
            h = self.node_mlp(torch.cat([mi, h], dim=-1))
        h = self.pre_norm(h)
        h = torch.clamp(h, min=-10, max=10)
        #h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        #if torch.isnan(h).any():
        #    print('NaN found in h after node_mlp')
        #    print(h)
        # print(h.shape)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        if self.bi:
            output2=self.mamba2(h.unsqueeze(0)).squeeze(0)
            output3=self.mamba2(torch.flip(h,[0]).unsqueeze(0)).sequeeze(0)
            output=h+output2+output3
        else:
            output=self.mamba2(h.unsqueeze(0)).squeeze(0)
        if self.dropout:
            output=self.out_dropout(output)
        if self.mamba_mlp:
            output=self.mamba_merge_mlp(torch.cat([h,output],-1))

        return output, coord, edge_attr


class E_GCL_vel(E_GCL_mamba):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL_mamba.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        src, dst = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        mi,edge_feat = self.edge_model(h,edge_index, radial, edge_attr)
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
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)


        coord += self.coord_mlp_vel(h) * vel
        if node_attr is not None:
            h = self.node_mlp(torch.cat([mi, h, node_attr], dim=-1))
        else:
            h = self.node_mlp(torch.cat([mi, h], dim=-1))
        h = self.pre_norm(h)
        h = torch.clamp(h, min=-10, max=10)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        if self.bi:
            output2 = self.mamba2(h.unsqueeze(0)).squeeze(0)
            output3 = self.mamba2(torch.flip(h, [0]).unsqueeze(0)).squeeze(0)
            output = h + output2 + output3
        else:
            output = self.mamba2(h.unsqueeze(0)).squeeze(0)
        if self.dropout:
            output = self.out_dropout(output)
        if self.mamba_mlp:
            output = self.mamba_merge_mlp(torch.cat([h, output], -1))
        return output, coord, edge_attr




class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh()) #we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)