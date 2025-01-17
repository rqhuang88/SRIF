from args import get_args
from pprint import pprint
from collections import defaultdict
from FlowModels.networks import PointFlow
import os
import open3d as o3d
import potpourri3d as pp3d
import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm
import torch
import torch.nn as nn
    
def post_process(source, target, args):
    class_name = 'cross'

    source_mesh = o3d.io.read_triangle_mesh(f'./runs/{class_name}/final/{source}_{target}.off')
    target_mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_dir, f'{target}.obj'))
    source_verts = np.asarray(source_mesh.vertices)
    source_faces = np.asarray(source_mesh.triangles)
    target_verts = np.asarray(target_mesh.vertices)
    
    try:        
        pts_refine = refine_mesh(source_verts, source_faces, target_verts, use_geod=True)
    except:
        pts_refine = refine_mesh(source_verts, source_faces, target_verts, use_geod=False)
    
    save_mesh = o3d.geometry.TriangleMesh()
    save_mesh.vertices = o3d.utility.Vector3dVector(pts_refine.squeeze().detach().cpu().numpy())
    save_mesh.triangles = o3d.utility.Vector3iVector(source_faces)
    o3d.io.write_triangle_mesh(f'./runs/{class_name}/refine/{source}_{target}.off', save_mesh)
    
def evaluate_visual(model, source, target, args):
    class_name = 'cross'
    save_dir = os.path.dirname(args.resume_checkpoint)
    
    source_mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_dir, f'{source}.obj'))
    target_mesh = o3d.io.read_triangle_mesh(os.path.join(args.data_dir, f'{target}.obj'))
    source_verts = np.asarray(source_mesh.vertices)
    source_faces = np.asarray(source_mesh.triangles)
    target_verts = np.asarray(target_mesh.vertices)

    integration_times = torch.linspace(0, args.time_length, steps=5)

    out = model.sample(source_verts, integration_times)
    
    for i in tqdm(range(int(out.shape[0]))):
        pts = out[i].reshape(-1, 3)
        save_mesh = o3d.geometry.TriangleMesh()
        
        os.makedirs(f'./runs/{class_name}/inter/', exist_ok=True)
        os.makedirs(f'./runs/{class_name}/final/', exist_ok=True)
        os.makedirs(f'./runs/{class_name}/refine/', exist_ok=True)
        
        if i == 0:
            save_mesh.vertices = o3d.utility.Vector3dVector(pts.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(source_faces)
            o3d.io.write_triangle_mesh(f'./runs/{class_name}/refine/{source}_{target}_{i}.off', save_mesh)
        elif i < int(out.shape[0])-1 and i > 0: 
            save_mesh.vertices = o3d.utility.Vector3dVector(pts.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(source_faces)
            o3d.io.write_triangle_mesh(f'./runs/{class_name}/inter/{source}_{target}_{i}.off', save_mesh)
        else:
            save_mesh.vertices = o3d.utility.Vector3dVector(pts.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(source_faces)
            o3d.io.write_triangle_mesh(f'./runs/{class_name}/final/{source}_{target}.off', save_mesh)
        
        
    
def main(args):
    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)
    
    train_filename = args.train_list
    with open(train_filename, 'r') as file:
        for line in tqdm(file):
            source, target = line.strip().split()
            save_dir = os.path.join("checkpoints", args.log_name)
            save_dir_class = save_dir + f'/{source}_{target}/'
            
            if os.path.exists(os.path.join(save_dir_class, 'checkpoint-3999.pt')):
                args.resume_checkpoint = os.path.join(save_dir_class, 'checkpoint-3999.pt')
            else:
                continue
                
            print("Resume Path:%s" % args.resume_checkpoint)
            checkpoint = torch.load(args.resume_checkpoint)
            model.load_state_dict(checkpoint['model'])
            model.eval()

            with torch.no_grad():
                evaluate_visual(model, source, target, args)
                
            # post_process(source, target, args)
            
            # exit()


if __name__ == '__main__':    
    args = get_args()
    main(args)
