import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
#一个node的特征有两个部分 1.coordinate 要求在SE(3)-euquivariant 中进行update 2.features invariant 常规的MLP即可进行update
#EGCL层 对节点特征（h）节点坐标(coordinate) 边特征(edge_feature)进行更新
class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
     #hidden_nf 隐藏层的输入维度
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False, norm_diff=True, tanh=False, coords_range=1, norm_constant=0):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2 #输入的边特征的维度（默认是节点特征维度的2倍）
        self.attention = attention#注意力机制
        self.norm_diff = norm_diff#对坐标进行归一化
        self.tanh = tanh #函数tanh处理坐标信息，映射距离；该属性判断是否对坐标进行tanh处理
        self.norm_constant = norm_constant#坐标缩放
        edge_coords_nf = 1#边的坐标维度

        #边MLP层
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        #节MLP层
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        #边特征更新的输出层，维度为1
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #边特征MLP层，映射到距离（边特征计算出来的距离）
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range #coords_range 坐标范围，边特征计算出来的距离tanh以后的缩放系数

        self.coord_mlp = nn.Sequential(*coord_mlp)
        #自注意力层
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
    #更新边特征
    #source,target 起始节点特征，终止节点特征
    #edge_attr,edge_mask 边特征，边编码
    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        #通过MLP层更新的边的新特征
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val #如果有使用注意力，通过自注意力机制层更新边的特征
        #边掩码处理
        if edge_mask is not None:
            out = out * edge_mask
        return out
    #更新节点特征 注意这里的x是指feature 不是coordinate node_attr是指额外的节点特征，默认为None
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index #获得节点source,target的编号
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))#聚合 将边信息聚合到节点中
        #合并node feature+聚合的边信息+node_attr
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        #update节点的特征
        out = x + self.node_mlp(agg)
        #return 更新后的节点特征和聚合的边信息
        return out, agg
   #更新坐标coordinate coor_diff 两个node的距离
    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        row, col = edge_index
        #边特征映射到距离，使用tanh后进行缩放---对节点的坐标用距离（标量）来更新，保证了输入的coordinate的equivariant
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        if edge_mask is not None:
            trans = trans * edge_mask
        #聚合到起始节点
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        #更新起始节点坐标
        coord = coord + agg
        return coord
    #input feature，coordinate，边的索引，边特征，额外节点特征 以及边和节点的编码
    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr   #获得更新的边特征，节点坐标，节点特征
    #计算原子间的距离 coordinate是坐标
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + self.norm_constant)

        return radial, coord_diff


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum', norm_constant=0, inv_sublayers=1, sin_embedding=False):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers #更新节点特征的卷积层数
        self.coords_range_layer = float(coords_range)/self.n_layers #坐标tanh后的缩放
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        #h,edge_feature,coordinate更新
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer, norm_constant=norm_constant))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        #h嵌入隐藏层中
        h = self.embedding(h)
        #hidden层中对h,x进行equivariant,invariant的更新
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        #从hidden层中输出更新的h
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


class EGNN_old(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum'):
        super(EGNN_old, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, attention=attention, norm_diff=norm_diff, tanh=tanh, coords_range=self.coords_range_layer))

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4,
                 attention=False, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        #GCL-graph convoluntional layer 更新invariant variables
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                              act_fn=act_fn, attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask,
                                               edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h



class TransformerNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, recurrent=True, attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, agg='sum', norm_constant=0):
        super(EGNN, self).__init__()
        hidden_initial = 128
        initial_mlp_layers = 1
        hidden_final = 128
        final_mlp_layers = 2
        n_heads = 8
        dim_feedforward = 512
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.initial_mlp = MLP(in_node_nf, hidden_nf, hidden_initial, initial_mlp_layers, skip=1, bias=True)

        self.decoder_layers = nn.ModuleList()
        if self.use_bn:
            self.bn_layers = nn.ModuleList()
        for _ in n_layers:
            self.decoder_layers.append(nn.TransformerEncoderLayer(hidden_nf, n_heads, dim_feedforward, dropout=0))

        self.final_mlp = MLP(hidden_nf, out_node_nf, hidden_final, final_mlp_layers, skip=1, bias=True)

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        """ x: batch_size, n, channels
            latent: batch_size, channels2. """
        x = F.relu(self.initial_mlp(x))
        for i in range(len(self.decoder_layers)):
            out = F.relu(self.decoder_layers[i](x))              # [bs, n, d]
            if self.use_bn and type(x).__name__ != 'TransformerEncoderLayer':
                out = self.bn_layers[i](out.transpose(1, 2)).transpose(1, 2)  # bs, n, hidden
            x = out + x if self.res and type(x).__name__ != 'TransformerEncoderLayer' else out
        return x


        # Edit Emiel: Remove velocity as input
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)



        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

    class SetDecoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            hidden, hidden_final = cfg.hidden_decoder, cfg.hidden_last_decoder
            self.use_bn = cfg.use_batch_norm
            self.res = cfg.use_residual
            self.cosine_channels = cfg.cosine_channels
            self.initial_mlp = MLP(cfg.set_channels,
                                   cfg.hidden_decoder,
                                   cfg.hidden_initial_decoder,
                                   cfg.initial_mlp_layers_decoder,
                                   skip=1, bias=True)

            self.decoder_layers = nn.ModuleList()
            if self.use_bn:
                self.bn_layers = nn.ModuleList()
            for layer in cfg.decoder_layers:
                self.decoder_layers.append(create_layer(layer, hidden, hidden, cfg))
                if self.use_bn:
                    self.bn_layers.append(nn.BatchNorm1d(hidden))

        def forward(self, x, latent):
            """ x: batch_size, n, channels
                latent: batch_size, channels2. """
            x = F.relu(self.initial_mlp(x, latent[:, self.cosine_channels:].unsqueeze(1)))
            for i in range(len(self.decoder_layers)):
                out = F.relu(self.decoder_layers[i](x))  # [bs, n, d]
                if self.use_bn and type(x).__name__ != 'TransformerEncoderLayer':
                    out = self.bn_layers[i](out.transpose(1, 2)).transpose(1, 2)  # bs, n, hidden
                x = out + x if self.res and type(x).__name__ != 'TransformerEncoderLayer' else out
            return x


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, width: int, nb_layers: int, skip=1, bias=True):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: hidden width
            nb_layers: number of layers
            skip: jump from residual connections
            bias: indicates presence of bias
        """
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.width = width
        self.nb_layers = nb_layers
        self.hidden = nn.ModuleList()
        self.lin1 = nn.Linear(self.dim_in, width, bias)
        self.skip = skip
        self.residual_start = dim_in == width
        self.residual_end = dim_out == width
        for i in range(nb_layers-2):
            self.hidden.append(nn.Linear(width, width, bias))
        self.lin_final = nn.Linear(width, dim_out, bias)

    def forward(self, x: Tensor):
        out = self.lin1(x)
        out = F.relu(out) + (x if self.residual_start else 0)
        for layer in self.hidden:
            out = out + layer(F.relu(out))
        out = self.lin_final(F.relu(out)) + (out if self.residual_end else 0)
        return out