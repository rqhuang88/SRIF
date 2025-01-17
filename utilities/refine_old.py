import argparse
import yaml
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
from zmq import device

from utilities.utils import chamfer_dist, compute_geodesic_distmat, search_t
from utilities.lib.deformation_graph import DeformationGraph


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def refine_mesh(source_mesh_V, source_mesh_F, target_mesh_V):
    global device
    use_geod = True
    dg = DeformationGraph()
    source_mesh_F, target_mesh_V = torch.from_numpy(source_mesh_F), torch.from_numpy(target_mesh_V)

    if use_geod:
        geod = compute_geodesic_distmat(source_mesh_V.detach().cpu().numpy(), source_mesh_F.detach().cpu().numpy())      
        dg.construct_graph(source_mesh_V.detach().cpu(), source_mesh_F, geod, device)
    else:
        geod = torch.cdist(source_mesh_V.detach().cpu().numpy(), source_mesh_V.detach().cpu().numpy(), p=2.0).cpu().numpy()
        dg.construct_graph_euclidean(source_mesh_V.detach().cpu(), geod, device)
    num_nodes = dg.nodes_idx.shape[0]

    opt_d_rotations = torch.zeros((1, num_nodes, 3)).to(device) # axis angle 
    opt_d_translations = torch.zeros((1, num_nodes, 3)).to(device)
    opt_d_rotations.requires_grad = True
    opt_d_translations.requires_grad = True
    surface_opt_params = [opt_d_rotations, opt_d_translations] 
    surface_optimizer = torch.optim.Adam(surface_opt_params, lr=0.005, betas=(0.9, 0.999))

    source_mesh_V = source_mesh_V.to(device)
    target_mesh_V = target_mesh_V.to(device)

    count = 0
    idx = 0
    eps = 1e-8

    with torch.no_grad():
        T12_pred = search_t(source_mesh_V, target_mesh_V) 
        T21_pred = search_t(target_mesh_V, source_mesh_V)
    
    while True:
        S_idx = torch.arange(0, source_mesh_V.shape[0]).to(device)
        S_temp1 = S_idx[T21_pred.squeeze()]
        S_temp2 = S_temp1[T12_pred.squeeze()]   
        Geo_dist_idx = geod[S_idx.cpu().numpy(), S_temp2.cpu().numpy()] < 0.01
        Geo_dist_idx = torch.from_numpy(Geo_dist_idx).to(device)

        new_source_mesh_V, arap, sr_loss = dg(source_mesh_V, opt_d_rotations, opt_d_translations)

        ## new source mesh
        target_mesh_V_T = target_mesh_V.squeeze()[T12_pred.squeeze()]
        target_mesh_V_nodes = target_mesh_V_T[Geo_dist_idx>0]
        new_source_mesh_V_nodes = new_source_mesh_V[Geo_dist_idx>0]
        
        ## loss
        cd_loss = chamfer_dist(new_source_mesh_V.unsqueeze(0), target_mesh_V.float().unsqueeze(0))
        loss_ali = F.mse_loss(new_source_mesh_V_nodes, target_mesh_V_nodes.float())

        loss = 0.01*loss_ali+arap+cd_loss+0.2*sr_loss

        if idx == 0:
            last_loss = torch.tensor([0]).to(device)
        elif abs(last_loss.item() - loss.item()) > eps:
            last_loss = loss.clone()
        else:
            count += 1
            if count > 15:
                return new_source_mesh_V

        if idx % 100 == 0:
            with torch.no_grad():
                T12_pred = search_t(new_source_mesh_V, target_mesh_V) 
                T21_pred = search_t(target_mesh_V, new_source_mesh_V) 

        idx = idx + 1
        surface_optimizer.zero_grad()
        loss.backward()
        surface_optimizer.step()
