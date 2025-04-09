import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from mamba_ssm import Mamba
import random

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention,self).__init__()
        self.multihead_attn=nn.MultiheadAttention(embed_dim,num_heads)
        self.linear=nn.Linear(embed_dim,embed_dim)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self, h1, h2):
        #h1 shape:(seq_len1,batch_size,embed_dim)
        #h2 shape:(seq_len2,batch_size,embed_dim)

        #Cross Attention from h1 to h2
        attn_output_h1,_=self.multihead_attn(query=1,key=h2,value=h2)
        #from h2 to h1
        attn_output_h2,_=self.multihead_attn(query=h2,key=h1,value=h1)
        #combine the outputs
        combined_output=attn_output_h1+attn_output_h2
        #pass through a linear layer and normalization
        output=self.linear(combined_output)
        output=self.norm(output)

#加入mamba处理节点特征的更新(替代原本EGNN_node_MLP作为全局更新器)，实现更有效的信息融合
class Mamba_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False,
                 bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        super(Mamba_GCL, self).__init__()
        input_edge = input_nf * 2 #输入边的特征维度（默认是输入节点特征维度的两倍)
        self.normalization_factor = normalization_factor#归一化的因数
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.bi=bi #双向，是否启用双向Mamba，用于更丰富的特征聚合
        self.d_state=d_state #Mamba的SSM维度，影响记忆长度
        self.dropout=dropout #控制mamba输出后的dropout概率，提升model泛化性
        self.mamba_mlp=mamba_mlp #控制是否使用mamba替换原始的MLP
        self.order_method=order_method
        if self.mamba_mlp:
            self.mamba_merge_mlp=nn.Sequential(
                 nn.Linear(hidden_nf+input_nf+nodes_att_dim,hidden_nf)
            )

        self.mamba2=Mamba(d_model=hidden_nf, #model dimension
                          d_state=64, #ssm state expansion factor
                          d_conv=4, #local convolution with
                          expand=1, #block expansion factor
                          )
        #边mlp层
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.edge_inf=nn.Sequential(nn.Linear(hidden_nf,1),nn.Sigmoid()) #calculate edge_w
        #节点mlp层
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        #注意力层
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        if self.dropout!=0:
            self.out_dropout=nn.Dropout(p=self.dropout)
   #更新边特征
    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        #更新后的边特征
        mij = self.edge_mlp(out)
        #边权重的存在概率(取值0~1），作为加权因子
        eij=self.edge_inf(mij)
        #out为经过注意力更新后的边特征 different from mij
        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij,eij
    #更新节点特征 这里的x是节点特征不是coordinate
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        #返回update的h和聚合的边信息
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        if torch.isnan(h).any():
            print("NaN found in input h")
            print(h)
        if torch.isnan(edge_attr).any():
            print("NaN found in input edge_attr")
            print(edge_attr)
        row, col = edge_index
        edge_feat, mij,eij= self.edge_model(h[row], h[col], edge_attr, edge_mask)
        #scatter_sum 使用 eij 作为加权因子，确保更重要的边对目标节点特征的贡献更大
        mi=scatter_sum(mij*eij,col,dim=0,dim_size=h.shape[0])
        if self.order_method=='degree':
            print('-----------------DEGREE!!!!!!!---------------------')
            degrees=torch.zeros(len(h),dtype=torch.int32).cuda()
            degrees.scatter_add_(0,row,torch.ones(row.size(0),dtype=torch.int32).cuda())
            degrees.scatter_add_(0,col,torch.ones(col.size(0),dtype=torch.int32).cuda())
            sorted_indices=torch.argsort(degrees)
            sorted_h=h[sorted_indices]
            h=sorted_h
            sorted_mi=mi[sorted_indices]
            mi=sorted_mi
        elif self.order_method == 'degree_with_shuffle':
            print('--------------DGREE WITH SHUFFLE---------------------')
            degrees = torch.zeros(len(h), dtype=torch.int32).cuda()
            degrees.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.int32).cuda())
            degrees.scatter_add_(0, col, torch.ones(col.size(0), dtype=torch.int32).cuda())
            sorted_indices=torch.argsort(degrees)
            unique_degrees=degrees[sorted_indices].unique(sorted==True)
            new_indices=[]
            for degree in unique_degrees:
                same_degree_indices=sorted_indices[degree[sorted_indices]==degree]
                same_degree_indices_list=same_degree_indices.tolist()
                random.shuffle(same_degree_indices_list)
                new_indices.extend(same_degree_indices_list)
            new_indices=torch.tensor(new_indices).cuda()
            sorted_h=h[new_indices]
            h=sorted_h
            sorted_mi=mi[new_indices]
            mi=sorted_mi
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        h=self.node_mlp(torch.cat([mi,h],-1))
        ######
        # if torch.isnan(h).any():
        #     print('NaN found in h after node_model')
        #     print(h)
        ######
        if node_mask is not None:
            h = h * node_mask
        #bi=Flase 单向模式 直接处理h；bi=True 双向模式 除了正向特征，还计算输入特征的反转（通过 torch.flip），从而获取更丰富的context
        if self.bi:
            output2=self.mamba2(h.unsqueeze(0)).squeeze(0)
            output3=self.mamba2(torch.filp(h,[0]).unsqueeze(0)).squeeze(0)
            # output = self.bi_mamba_merge_mlp(torch.cat([output1, output2, output3], -1))
            output=h+output2+output3
        else:
            output=self.mamba2(h.unsqueeze(0)).squeeze(0)
            #if torch.isnan(output).any():
            #    print("NaN found in output")
            #    print(output)
            #output=self.mamba_merge_mlp(torch.cat([output1, output2], -1))
            #output=output1+output2
        if self.dropout:
            output=self.out_dropout(output)
        #   if torch.isnan(output).any():
        #       print("NaN found in output after dropout")
        #       print(output)
        # output = self.cross_attention(h, output)
        if self.mamba_mlp:
            output = self.mamba_merge_mlp(torch.cat([h, output], -1))
            #if torch.isnan(output).any():
            #    print("NaN found in output after mamba_merge_mlp")
            #    print(output)

        return output

#等变(equivariant)操作--update node coordinate
class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh #对坐标进行处理
        self.coords_range = coords_range #缩变系数
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)#边更新的输出层，维度1
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #coordinate的mlp，更新边的特征，映射到距离（由边的特征计算出来）
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
   #更新coordinate 符合SE3-euqivariant的特性，从而EquivariantUpdate层符合SE3-equivariant
    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        #边特征映射到距离，使用tanh后进行缩放
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

#等变模块 input:h,x,edge_index,node_mask,edge_mask,edge_attr output:updated h,x
class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',
                 bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding #距离正弦余弦嵌入
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bi = bi
        self.d_state = d_state
        self.dropout = dropout
        self.order_method=order_method
        self.mamba_mlp=mamba_mlp
        for i in range(0, n_layers):
            #h
            self.add_module("mamba_gcl_%d" % i, Mamba_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method,
                                              bi=self.bi,d_state=self.d_state,dropout=self.dropout,order_method=self.order_method,mamba_mlp=self.mamba_mlp))
            #x
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        #计算边的两个node之间的距离
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        #距离进行正弦/余弦嵌入
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        #将node之间的distance作为边的feature，contact其他边特征
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        #update h,x
        for i in range(0, self.n_layers):
            h, _ = self._modules["mamba_gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGMN(nn.Module):
    #init中包含了h的input，hidden，output的维度转化，self.embedding初始嵌入层，self.embedding_out最后的输出层
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        super(EGMN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers#更新层数
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bi = bi
        self.d_state = d_state
        self.dropout = dropout
        self.order_method = order_method
        self.mamba_mlp = mamba_mlp
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        #多个EquivariantBlock组成的EGNN
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               bi=self.bi,d_state=self.d_state,dropout=self.dropout,order_method=self.order_method,mamba_mlp=self.mamba_mlp))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class GMN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None,
                 bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.bi = bi
        self.d_state = d_state
        self.dropout = dropout
        self.order_method = order_method
        self.mamba_mlp = mamba_mlp
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("mamba_gcl_%d" % i, Mamba_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention,bi=bi,order_method=order_method,d_state=d_state,dropout=dropout,mamba_mlp=mamba_mlp))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["mamba_gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h