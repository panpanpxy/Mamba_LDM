import torch
import torch.nn as nn
from egnn.egnn_mamba import EGMN,GMN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
class EGMN_dynamics_QM9(nn.Module):
    def __init__(self,in_node_nf,context_node_nf,
                 n_dims,hidden_nf=64,device='cpu',
                 act_fn=torch.nn.SiLU(),n_layers=4,attention=False,
                 condition_time=True,tanh=False,mode='egmn_dynamics',norm_constant=0,
                 inv_sublayers=2,sin_embedding=False,normalization_factor=100,aggregation_method='sum',
                 bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        super().__init__()
        self.mode=mode
        if mode=='egmn_dynamics':
            self.egmn=EGMN(n_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                bi=bi, d_state=d_state, dropout=dropout, order_method=order_method,
                mamba_mlp=mamba_mlp)
            self.in_node_nf=in_node_nf
        elif mode == 'gmn_dynamics':
            self.gmn = GMN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method,
                i=bi, d_state=d_state, dropout=dropout, order_method=order_method,
                mamba_mlp=mamba_mlp)
        self.context_node_nf=context_node_nf
        self.device=device
        self.n_dims=n_dims
        self._edge_dict={}
        self.condition_time=condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward
        # =def forward but_表示内部实现的辅助方法，非直接对外接口--内部方法，用于封装核心forward逻辑，用户可通过外部接口（warp_forward/forward)调用，然后间接调用_forward
        # xh-h和x的组合tensor shape:(batch_size,n_nodes,dims)

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape  # bs是指batch_size 批次大小
        h_dims = dims - self.n_dims  # 特征维度，意思是总维度除去空间维度（x的dim)
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()  # get position
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()  # 获取特征信息

        if self.condition_time:  # 如果True,即启用condition_time，将时间步长t作为额外的feature add到node_feature中
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:  # 若context存在，作为额外信息参与节点特征的更新
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            # add context来更新h
            h = torch.cat([h, context], dim=1)
        # egmn or gmn来更新h,x(message passing) vel计算速度矢量，即位置更新的变化量
        if self.mode == 'egmn_dynamics':
            h_final, x_final = self.egmn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gmn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gmn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:  # 使用mask移除速度的均值，确保平移不影响结果
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)
        # 生成每个批次图的edge连接信息 GNN中，用edge_index表示邻接关系，而不是显示存储整个邻接矩阵
        # 生成的edges用于self.egnn中，指定了node之间的信息传递路径，node的特征通过edge进行agg和update

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:  # 从缓存字典中读取n_nodes
            edges_dic_b = self._edges_dict[n_nodes]  # batch_size[]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample 生成边的索引rows,cols
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]  # return 两个tensor（row,col)，用于描述图的连接关系
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

class EGMN_encoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egmn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True,bi = False, order_method = 'No', d_state = 64, dropout = 0, mamba_mlp = False):
        '''

        Parameters
        ----------
        in_node_nf: Number of invariant features for input nodes.

        '''
        super().__init__()
        include_charges=int(include_charges)
        num_classes=in_node_nf-include_charges #h_cat
        self.mode=mode
        if mode=='egmn_dynamics':
            self.egmn = EGMN(n_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                             hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                             n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                             inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                             normalization_factor=normalization_factor,
                             aggregation_method=aggregation_method,
                             bi=bi, d_state=d_state, dropout=dropout, order_method=order_method,
                             mamba_mlp=mamba_mlp)
            self.in_node_nf = in_node_nf
        elif mode == 'gmn_dynamics':
           self.gmn = GMN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=hidden_nf + 3,
                in_edge_nf=0, hidden_nf=hidden_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method,
               bi=bi, d_state=d_state, dropout=dropout, order_method=order_method,
               mamba_mlp=mamba_mlp
           )




        self.final_mlp=nn.Sequential(
            nn.Linear(hidden_nf,hidden_nf),
            act_fn,
            nn.Linear(hidden_nf,out_node_nf*2+1)
        )
        self.num_classes=num_classes
        self.include_charges=include_charges
        self.context_node_nf=context_node_nf
        self.device=device
        self.n_dims=n_dims
        self._edge_dict={}
        #self.condition_time=condition_time
        self.out_node_nf=out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egmn_dynamics':
            h_final, x_final = self.egmn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gmn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gmn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)
        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGMN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel=remove_mean(vel)
        else:
            vel=remove_mean_with_mask(vel,node_mask.view(bs,n_nodes,1))

        h_final=self.final_mlp(h_final)
        h_final=h_final*node_mask if node_mask is not None else h_final
        h_final=h_final.view(bs,n_nodes,-1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1:1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])
        # calculate mean and std 为了后面可以定义高斯分布(对应mean=mu sigma=std)，用于sample和计算KL divergence
        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one.')
            vel_std = torch.ones_like(vel_std)

        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one.')
            h_std = torch.ones_like(h_std)

        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization
        # vel_mean=z_x_mu vel_std=z_x_sigma;h_mean=z_h_mu h_std=z_h_sigma

        return vel_mean, vel_std, h_mean, h_std
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

class EGMN_decoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egmn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True, bi=False, order_method='No', d_state=64, dropout=0, mamba_mlp=False):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges
        if mode == 'egmn_dynamics':
            self.egmn = EGMN(n_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                             hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                             n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                             inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                             normalization_factor=normalization_factor,
                             aggregation_method=aggregation_method,
                             bi=bi, d_state=d_state, dropout=dropout, order_method=order_method,
                             mamba_mlp=mamba_mlp)
            self.in_node_nf = in_node_nf
        elif mode == 'gmn_dynamics':
            self.gmn = GMN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method,
                i=bi, d_state=d_state, dropout=dropout, order_method=order_method,
                mamba_mlp=mamba_mlp)
        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        # self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egmn_dynamics':
            h_final, x_final = self.egmn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gmn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gmn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)
        # vel--x_recon, h_final--h_recon
        return vel, h_final

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

