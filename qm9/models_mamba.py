import torch
from torch.distributions.categorical import Categorical

import numpy as np
from egnn.model_mamba import EGMN_dynamics_QM9, EGMN_encoder_QM9, EGMN_decoder_QM9

from equivariant_diffusion.en_mamba_diffusion import EnVariationalDiffusion, EnHierarchicalVAE, EnLatentDiffusion
import pickle
from os.path import join

from qm9.models import DistributionNodes, DistributionProperty


def get_model(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGMN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        bi=args.bi, d_state=args.d_state, dropout=args.dropout, order_method=args.order_method,
        mamba_mlp=args.mamba_mlp
    )

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_autoencoder(args, device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    # if args.condition_time:
    #     dynamics_in_node_nf = in_node_nf + 1
    # else:
    print('Autoencoder models are _not_ conditioned on time.')
    # dynamics_in_node_nf = in_node_nf

    encoder = EGMN_encoder_QM9(
        in_node_nf=in_node_nf, context_node_nf=args.context_node_nf, out_node_nf=args.latent_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=1,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        include_charges=args.include_charges,bi=args.bi, d_state=args.d_state, dropout=args.dropout, order_method=args.order_method,
        mamba_mlp=args.mamba_mlp
    )

    decoder = EGMN_decoder_QM9(
        in_node_nf=args.latent_nf, context_node_nf=args.context_node_nf, out_node_nf=in_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        include_charges=args.include_charges,bi=args.bi, d_state=args.d_state, dropout=args.dropout, order_method=args.order_method,
        mamba_mlp=args.mamba_mlp
    )

    vae = EnHierarchicalVAE(
        encoder=encoder,
        decoder=decoder,
        in_node_nf=in_node_nf,
        n_dims=3,
        latent_node_nf=args.latent_nf,
        kl_weight=args.kl_weight,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges
    )

    return vae, nodes_dist, prop_dist


def get_latent_diffusion(args, device, dataset_info, dataloader_train):
    '''
    get GeoLDM(VAE+SE3-equivariant)

    '''
    # Create (and load) the first stage model (Autoencoder).
    if args.ae_path is not None:
        with open(join(args.ae_path, 'args.pickle'), 'rb') as f:
            first_stage_args = pickle.load(f)
    else:
        first_stage_args = args

    # CAREFUL with this -->
    # 判断first_stage_args参数中，是否包含normalization_factor & aggregation_method属性
    if not hasattr(first_stage_args, 'normalization_factor'):
        first_stage_args.normalization_factor = 1
    if not hasattr(first_stage_args, 'aggregation_method'):
        first_stage_args.aggregation_method = 'sum'

    device = torch.device("cuda" if first_stage_args.cuda else "cpu")
    # 实例化AE model,latent向量维度为latent_nf
    first_stage_model, nodes_dist, prop_dist = get_autoencoder(
        first_stage_args, device, dataset_info, dataloader_train)
    first_stage_model.to(device)
    # 如果AE model已经训练过(args.ae_path is Not None),load paramters of AE model
    if args.ae_path is not None:
        fn = 'generative_model_ema.npy' if first_stage_args.ema_decay > 0 else 'generative_model.npy'
        flow_state_dict = torch.load(join(args.ae_path, fn),
                                     map_location=device)
        # first_stage_args.load_state_dict(flow_state_dict)---根据VAE模型的权重文件是否存在，load/expampify VAE model
        first_stage_model.load_state_dict(flow_state_dict)

    # Create the second stage model (Latent Diffusions).
    # AE model output--latent_nf,as input of Latent Diffusion
    args.latent_nf = first_stage_args.latent_nf
    in_node_nf = args.latent_nf
    # 看看要不要考虑condition_time?
    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf
    # 实例化SE3-EGMN模型
    net_dynamics = EGMN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
        bi=args.bi, d_state=args.d_state, dropout=args.dropout, order_method=args.order_method,
        mamba_mlp=args.mamba_mlp
    )
    # 基于之前实例好的VAE+SE3-EGNN模型，创建GeoLDM
    if args.probabilistic_model == 'diffusion':
        vdm = EnLatentDiffusion(
            vae=first_stage_model,  # VAE
            trainable_ae=args.trainable_ae,
            dynamics=net_dynamics,  # SE3-EGNN
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )

        return vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)

def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim

if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
