import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.distributed
import numpy as np
from tqdm import tqdm
import open3d as o3d
import random
import faulthandler
import pymeshlab
import torch.multiprocessing as mp
import time
import scipy.misc
from FlowModels.networks import PointFlow
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, apply_random_rotation, save, resume, visualize_point_clouds
from tensorboardX import SummaryWriter
from utilities.utils import chamfer_dist, save_pointcloud_as_ply, load_point_clouds_process, construct_arap, compute_arap

faulthandler.enable()


def farthest_point_sample(xyz, npoint):
    N, C = xyz.shape
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.full((N,), np.inf, dtype=np.float32)
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid)**2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


def main_worker(gpu, source, target, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name + f'/{source}_{target}/'
    else:
        log_dir = "runs/time-%d" % time.time() + f'/{source}_{target}/'
    os.makedirs(log_dir, exist_ok=True)
        
    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None

    try:
        source_mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_dir, f'{source}.obj'))
        target_mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_dir, f'{target}.obj'))
        source_verts = np.asarray(source_mesh.vertices)
        source_faces = np.asarray(source_mesh.triangles)
        target_verts = np.asarray(target_mesh.vertices)
        
        one_ring_neigh = construct_arap(source_verts, source_faces)
    except:
        source_mesh = o3d.io.read_point_cloud(os.path.join(args.data_dir, f'{source}.ply'))
        target_mesh = o3d.io.read_point_cloud(os.path.join(args.data_dir, f'{target}.ply'))
        source_verts = np.asarray(source_mesh.points)
        target_verts = np.asarray(target_mesh.points)
        
        one_ring_neigh = construct_arap(source_verts, None)
    
    inter_pcd_frame = load_point_clouds_process(os.path.join(args.inter_pcd, f'{source}_{target}'), source_verts, save_fps=False)
    inter_pcd_frame = torch.from_numpy(inter_pcd_frame).float().cuda()
    
    model = PointFlow(args)
    if args.distributed:  
        if args.gpu is not None:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.gpu], output_device=args.gpu, check_reduction=True)

            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.multi_gpu_wrapper(_transform_)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
        else:
            assert 0, "DistributedDataParallel constructor should always set the single device scope"
    elif args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:  # Single process, multiple GPUs per process
        def _transform_(m):
            return nn.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    # resume checkpoints
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    start_time = time.time()
    inter_avg_meter = AverageValueMeter()
    point_avg_meter = AverageValueMeter()
    arap_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        model.train()
        integration_times = None
        out = model(source_verts, target_verts, inter_pcd_frame, one_ring_neigh, optimizer, epoch, integration_times, writer)
        inter_loss, recon_loss, arap_loss = out['inter'], out['recon'], out['arap']
        inter_avg_meter.update(inter_loss)
        point_avg_meter.update(recon_loss)
        arap_avg_meter.update(arap_loss)
        if epoch % args.log_freq == 0:
            duration = time.time() - start_time
            start_time = time.time()
            print("[Rank %d] Epoch %d InterLoss %2.5f PointLoss %2.5f ArapLoss %2.5f"
                  % (args.rank, epoch, inter_avg_meter.avg, point_avg_meter.avg, arap_avg_meter.avg))


        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq == 0:
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-latest.pt'))


def main():
    args = get_args()
    save_dir = os.path.join("checkpoints", args.log_name)

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)
    
    ngpus_per_node = torch.cuda.device_count()
    
    train_filename = args.train_list
    
    with open(train_filename, 'r') as file:
        for line in tqdm(file):
            source, target = line.strip().split()
            save_dir_class = save_dir + f'/{source}_{target}/'
            
            os.makedirs(save_dir_class, exist_ok=True)
            with open(os.path.join(save_dir_class, 'command.sh'), 'w') as f:
                f.write('python -X faulthandler ' + ' '.join(sys.argv))
                f.write('\n')
                
            if args.distributed:
                args.world_size = ngpus_per_node * args.world_size
                mp.spawn(main_worker, nprocs=ngpus_per_node, args=(source, target, save_dir_class, ngpus_per_node, args))
            else:
                main_worker(args.gpu, source, target, save_dir_class, ngpus_per_node, args)
    
if __name__ == '__main__':
    main()
