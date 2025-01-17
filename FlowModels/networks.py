import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from FlowModels.flow import get_point_cnf
from utils import truncated_normal, reduce_tensor, standard_normal_logprob
from utilities.utils import chamfer_dist, compute_arap

# Model
class PointFlow(nn.Module):
    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.recon_weight = args.recon_weight
        self.distributed = args.distributed
        self.point_cnf = get_point_cnf(args)

    def multi_gpu_wrapper(self, f):
        self.point_cnf = f(self.point_cnf)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.point_cnf.parameters()))
        return opt

    def forward(self, source_verts, target_verts, inter_pcd_frame, one_ring_neigh, opt, step, integration_times=None, writer=None):
        opt.zero_grad()
        source_pcd = torch.from_numpy(source_verts).float().to('cuda')
        target_pcd = torch.from_numpy(target_verts).float().to('cuda')
        one_ring_neigh = one_ring_neigh.to('cuda')
        x = source_pcd
        num_points = x.size(0)
        
        # Compute the reconstruction likelihood P(X|z)
        y, _ = self.point_cnf(x, torch.zeros(num_points, 1).to(x), integration_times=integration_times, reverse=True)    

        # Loss
        y_inital, y_inter, y_final = y[0], y[1:5], y[5]
        inital_loss = chamfer_dist(y_inital.unsqueeze(0), source_pcd.unsqueeze(0))
        inter_loss = chamfer_dist(y_inter, inter_pcd_frame)+10*inital_loss
        recon_loss = chamfer_dist(y_final.unsqueeze(0), target_pcd.unsqueeze(0))
        arap_loss = compute_arap(y, one_ring_neigh)
        
        loss = inter_loss+10*recon_loss+2*arap_loss

        loss.backward()
        opt.step()

        if writer is not None:
            writer.add_scalar('train/inter', inter_loss, step)
            writer.add_scalar('train/recon', recon_loss, step)
            writer.add_scalar('train/arap', arap_loss, step)

        return {
            'inter': inter_loss,
            'recon': recon_loss,
            'arap': arap_loss,
        }


    def sample(self, source_verts, integration_times=None):
        x = torch.from_numpy(source_verts).float().to('cuda')
        y = self.point_cnf(x, integration_times=integration_times, reverse=True)
        return y

