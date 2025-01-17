from typing import Union
import random
import numpy as np
import torch
import torch.nn as nn
import trimesh
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import KDTree
import scipy.linalg as linalg
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import pymeshlab
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras, FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader, SoftPhongShader, HardFlatShader,
    PointLights, Textures, blending, HardGouraudShader, Materials
)
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.io import IO


def get_vert_connectivity(mesh_v, mesh_f):
    row = np.append(mesh_f[:,0], mesh_f[:,1])
    row = np.append(row, mesh_f[:,2])
    col = np.append(mesh_f[:,1], mesh_f[:,2])
    col = np.append(col, mesh_f[:,0])
    data = np.ones(len(row))
    ij = np.vstack((row, col))
    matrix = sp.coo_matrix((data, ij), shape=(len(mesh_v), len(mesh_v)))
    matrix = matrix + matrix.T
    return sp.csc_matrix(matrix)


def construct_arap(vert, face, max_neigh_num=18):
    if face is not None:
        adj_matrix = get_vert_connectivity(vert, face).tolil()
        one_ring_neigh = adj_matrix.rows
        one_ring_neigh = [np.pad(neigh, (0, max_neigh_num - len(neigh)), mode='constant', constant_values=i)
                          if len(neigh) < max_neigh_num else neigh[:max_neigh_num] for i, neigh in enumerate(one_ring_neigh)]
    else:
        neigh = NearestNeighbors(n_neighbors=max_neigh_num, algorithm='auto')
        neigh.fit(vert)
        distances, indices = neigh.kneighbors(vert)
        one_ring_neigh = [np.pad(neigh, (0, max_neigh_num - len(neigh)), mode='constant', constant_values=i)
                          if len(neigh) < max_neigh_num else neigh for i, neigh in enumerate(indices)]
    
    one_ring_neigh = torch.tensor(list(one_ring_neigh))
    return one_ring_neigh


def compute_arap(verts, one_ring_neigh, max_neigh_num=18):
    B, N, _ = verts.shape
    batch_indices = torch.arange(B, device=verts.device).view(B, 1, 1)
    batch_indices = batch_indices.expand(B, N, max_neigh_num)
    one_ring_neigh_expanded = one_ring_neigh.unsqueeze(0).expand(B, N, max_neigh_num)
    neighbor_verts = verts[batch_indices, one_ring_neigh_expanded, :]
    diff_term = verts.unsqueeze(2) - neighbor_verts
    arap_loss = torch.sum(diff_term ** 2) / (B * N)
    return arap_loss


def load_point_clouds_process(files_list, source_pointcloud, save_fps=True):
    point_clouds = []  # 存放读取的点云数据
    i, idx_use = 0, 0
    for file_path in tqdm(os.listdir(files_list)):
        last_name = ''.join(file_path).split('.')[-1]
        if last_name != 'ply':
            continue
        pcd_path = os.path.join(files_list, file_path)
        ms = pymeshlab.MeshSet()  # 创建一个新的 MeshSet
        ms.load_new_mesh(pcd_path)  # 加载点云文件
        vc = ms.current_mesh().vertex_matrix()  # 获取顶点坐标矩阵
        if save_fps and vc.shape[0] > source_pointcloud.shape[0]:
            vc = remove_outliers(vc)
            vc_filter = farthest_point_sample_np(vc, source_pointcloud.shape[0])
            # vc_filter = farthest_point_sample_np(vc, vc.shape[0] // 10)
        else:
            vc_filter = vc
        # if save_fps:
        #     if i == 0:
        #         idx = select_closest_points(source_pointcloud, vc)
        #         idx_use = idx
        #     vc_filter = vc[idx_use]
        # else:
        #     vc_filter = vc
    
        if save_fps:
            old_folder_path = Path(files_list)
            new_folder_path = old_folder_path.parent / (old_folder_path.name + '_fps')
            new_folder_path.mkdir(exist_ok=True)
            dst_file_path = './' + str(new_folder_path / file_path)
            save_pointcloud_as_ply(vc_filter, dst_file_path)
        
        point_clouds.append(vc_filter)  # 追加到列表
        i += 1
    num_points = [pc.shape[0] for pc in point_clouds]
    if len(set(num_points)) != 1:
        raise ValueError("Not all point clouds have the same number of points.")

    point_clouds_np = np.stack(point_clouds, axis=0)
    return point_clouds_np


def filter_surface_points(source_points, target_points, threshold):
    tree = KDTree(source_points)
    distances, _ = tree.query(target_points)
    surface_points = target_points[distances < threshold]
    return surface_points


def select_closest_points(source_points, target_points):
    nbrs = neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_points)
    distances, indices = nbrs.kneighbors(source_points)
    # closest_points = target_points[indices.flatten()]

    return indices.flatten()


def remove_outliers(pointcloud, k=10, threshold_scale=2):
    nbrs = neighbors.NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pointcloud)
    distances, indices = nbrs.kneighbors(pointcloud)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = np.mean(mean_distances) + threshold_scale * np.std(mean_distances)
    filtered_idx = mean_distances < threshold
    
    return pointcloud[filtered_idx]


def normalize_pointcloud(pointcloud):
    centroid = torch.mean(pointcloud, dim=0)
    pointcloud_centered = pointcloud - centroid
    max_distance = torch.max(torch.sqrt(torch.sum(pointcloud_centered ** 2, dim=1)))
    pointcloud_normalized = pointcloud_centered / max_distance

    return pointcloud_normalized


def save_pointcloud_as_ply(vertices, ply_file_path):
    try:
        vertices = vertices.detach().cpu().numpy()
    except:
        vertices = vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(ply_file_path, pcd)


def farthest_point_sample_np(points, num_samples):
    N, D = points.shape
    farthest_pts_idx = np.zeros(num_samples, dtype=np.int64)
    distances = np.ones(N) * np.inf
    farthest_pts_idx[0] = 0
    for i in range(1, num_samples):
        dist_to_last = np.sum(np.square(points - points[farthest_pts_idx[i-1]]), axis=1)
        distances = np.minimum(distances, dist_to_last)
        farthest_pts_idx[i] = np.argmax(distances)
    farthest_pts_pcd = points[farthest_pts_idx]
    return farthest_pts_pcd


def farthest_point_sample(xyz, npoint):
    xyz = torch.from_numpy(xyz).float()
    xyz = xyz.unsqueeze(0)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N,dtype=torch.float32).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        single_path=False,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    # print(x_mask.shape,x_mask.sum(), y_mask.sum())
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    if not single_path:            
        cham_dist = cham_x + cham_y
    else:
        cham_dist = 0*cham_x + cham_y

    return cham_dist


def arap_cost (R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R [:, None]
    g_i = g [:, None]
    t_i = t [:, None]

    g_j = g [e]
    t_j = t [e]

    if lietorch :
        e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    else :
        e_ij = (((R_i @ (g_j - g_i)[...,None]).squeeze() + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    
    o = (w * e_ij ).mean()

    return o


def projective_depth_cost(dx, dy):

    x_mask = dx> 0
    y_mask = dy> 0
    depth_error = (dx - dy) ** 2
    depth_error = depth_error[y_mask * x_mask]
    silh_loss = torch.mean(depth_error)

    return silh_loss

def silhouette_cost(x, y):

    x_mask = x[..., 0] > 0
    y_mask = y[..., 0] > 0
    silh_error = (x - y) ** 2
    silh_error = silh_error[~y_mask]
    silh_loss = torch.mean(silh_error)

    return silh_loss

def landmark_cost(x, y, landmarks):
    x = x [ landmarks[0] ]
    y = y [ landmarks[1] ]
    loss = torch.mean(
        torch.sum( (x-y)**2, dim=-1 ))
    return loss

def chamfer_dist(src_pcd, tgt_pcd, single_path=False):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    src=torch.randperm(src_pcd.shape[1])
    tgt=torch.randperm(tgt_pcd.shape[1])
    s_sample = src_pcd[:, src]
    t_sample = tgt_pcd[:, tgt]
    cham_dist = compute_truncated_chamfer_distance(s_sample, t_sample, trunc=10000, single_path=single_path)

    return cham_dist



def chamfer_dist_show(src_pcd,   tgt_pcd , trun):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    samples = 3000
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src]
    t_sample = tgt_pcd[ tgt]
    cham_dist = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=trun)

    return cham_dist


def differentiable_binarize(input_tensor, threshold=0.5, tau=1.0):
    sigmoid_tensor = torch.sigmoid((input_tensor - threshold) * tau)
    return sigmoid_tensor

def make_eyes_circle(eye, up, n):
    rot_matrix = linalg.expm(np.cross(np.eye(3), up / linalg.norm(up) * np.pi * 2 / n))
    rotation = torch.Tensor(rot_matrix)
    for i in range(n - 1):
        new_eye = torch.mv(rotation, eye[i])
        new_eye = new_eye.view([1, 3])
        eye = torch.cat([eye, new_eye], dim=0)
    return eye

def make_cameras(eyes, center, up, zfar=100, znear=1, fov=60, aspect_ratio=1, device=("cpu")):
    R, T = look_at_view_transform(eye=eyes, at=center, up=up, device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, znear=znear, aspect_ratio=aspect_ratio, fov=fov, device=device)
    return cameras

def render_mesh(mesh, cameras, H, W, device):
    rasterizer = MeshRasterizer(cameras, RasterizationSettings((H, W), faces_per_pixel=16))
    lights = PointLights(device=device, location=cameras.get_camera_center() + torch.Tensor([0,6,0]).to(device))
    # shader = HardFlatShader(device=device, cameras=cameras, lights=lights)
    shader = HardFlatShader(device=device, cameras=cameras, lights=lights, blend_params=blending.BlendParams(background_color=(0, 0, 0)))
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    images = renderer(mesh.extend(len(cameras)))
    return renderer, images

def render_mesh2image(verts, faces, device, convert_to_pointcloud=False):
    H = 512
    W = 512
    fov = 30
    center = torch.Tensor([[0, 0, 0]])  # look_at target
    eye = torch.Tensor([[0, 0, 3.5]])  # camera position
    up = torch.Tensor([[0, 1, 0]])  # camera orientation
    near_plane = 0.1
    far_plane = 50.0
    aspect_ratio = 1
    
    # Create a renderer with the desired image size
    verts, faces = verts.squeeze().to(device), faces.to(device)
    mesh = Meshes(verts=[verts], faces=[faces])
    eyes = make_eyes_circle(eye=eye, up=up, n=16)
    cameras = make_cameras(eyes=eyes, center=center, up=up, fov=fov, zfar=far_plane, znear=near_plane, aspect_ratio=aspect_ratio, device=device)
    vertices = mesh.verts_list()[0]
    # verts_rgb = torch.tensor((1, 1, 1), device=device).expand([vertices.shape[0], vertices.shape[1]]).view(1, -1, 3)
    # mesh.textures = Textures(verts_rgb=verts_rgb)
    mesh.textures = Textures(verts_rgb=torch.ones_like(vertices).view(1, -1, 3))
    _, image = render_mesh(mesh, cameras, H, W, device)
    
    return image


def knnsearch_t(x, y):
    distance = torch.cdist(x.float(), y.float())
    # distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx


def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    T21 = knnsearch_t(A2, A1)
    return T12


def ICP_rot(source, target, T12, T21, idx):
    target_T = target.squeeze()[T12.squeeze() -1 ]
    target_nodes = target_T[idx>0]
    source_nodes = source.squeeze()[idx>0]
    SS = torch.transpose(source_nodes, 1, 0).matmul(target_nodes)

    U, S, V = torch.svd(SS)
    R = V.matmul(torch.transpose(U,1,0))
    Target_new = target.matmul(R)

    return Target_new


def compute_vertex_normals(vertices, faces):
    """
    Computes the vertex normals of a mesh given its vertices and faces.
    vertices: a tensor of shape (num_vertices, 3) containing the 3D positions of the vertices
    faces: a tensor of shape (num_faces, 3) containing the vertex indices of each face
    returns: a tensor of shape (num_vertices, 3) containing the 3D normals of each vertex
    """
    # Compute the face normals
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]
    face_normals = torch.cross(p1 - p0, p2 - p0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)    # Accumulate the normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)    # Normalize the accumulated normals
    vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)

def compute_geodesic_distmat(verts, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm

    Args:
        verts (np.ndarray): array of vertices coordinates [n, 3]
        faces (np.ndarray): array of triangular faces [m, 3]

    Returns:
        geo_dist: geodesic distance matrix [n, n]
    """
    NN = 500

    # get adjacency matrix
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vertex_adjacency = mesh.vertex_adjacency_graph
    assert nx.is_connected(vertex_adjacency), 'Graph not connected'
    vertex_adjacency_matrix = nx.adjacency_matrix(vertex_adjacency, range(verts.shape[0]))
    # get adjacency distance matrix
    graph_x_csr = neighbors.kneighbors_graph(verts, n_neighbors=NN, mode='distance', include_self=False)
    distance_adj = csr_matrix((verts.shape[0], verts.shape[0])).tolil()
    distance_adj[vertex_adjacency_matrix != 0] = graph_x_csr[vertex_adjacency_matrix != 0]
    # compute geodesic matrix
    geodesic_x = shortest_path(distance_adj, directed=False)
    if np.any(np.isinf(geodesic_x)):
        print('Inf number in geodesic distance. Increase NN.')
    return geodesic_x

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    # print(pc.shape)
    # if pc.shape[0] > 5000:
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # pc = pc / m
    return pc

def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
        
class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)
    
def compute_vertex_normals(vertices, faces):
    """
    Computes the vertex normals of a mesh given its vertices and faces.
    vertices: a tensor of shape (num_vertices, 3) containing the 3D positions of the vertices
    faces: a tensor of shape (num_faces, 3) containing the vertex indices of each face
    returns: a tensor of shape (num_vertices, 3) containing the 3D normals of each vertex
    """
    # Compute the face normals
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]
    face_normals = torch.cross(p1 - p0, p2 - p0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)

    # Accumulate the normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)

    # Normalize the accumulated normals
    vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)

    return vertex_normals

def compute_face_normals(vertices, faces):
    """
    Compute the face normals for a given mesh.

    Args:
        vertices (torch.Tensor): The vertices of the mesh, shape (num_vertices, 3)
        faces (torch.Tensor): The faces of the mesh, shape (num_faces, 3)

    Returns:
        normals (torch.Tensor): The face normals, shape (num_faces, 3)
    """
    face_vertices = vertices[faces]  # shape (num_faces, 3, 3)
    edge_vectors = torch.roll(face_vertices, -1, dims=1) - face_vertices
    normals = torch.cross(edge_vectors[:, 0], edge_vectors[:, 1], dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return normals


def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def nn_interpolate(desc, xyz, dists, idx, idf):
    xyz = xyz.unsqueeze(0)
    B, N, _ = xyz.shape
    mask = torch.from_numpy(np.isin(idx.numpy(), idf.numpy())).int()
    mask = torch.argsort(mask, dim=-1, descending=True)[:, :, :3]
    dists, idx = torch.gather(dists, 2, mask), torch.gather(idx, 2, mask)
    transl = torch.arange(dists.size(1))
    transl[idf.flatten()] = torch.arange(idf.flatten().size(0))
    shape = idx.shape
    idx = transl[idx.flatten()].reshape(shape)
    dists, idx = dists.to(desc.device), idx.to(desc.device)

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_points = torch.sum(index_points(desc, idx) * weight.view(B, N, 3, 1), dim=2)

    return interpolated_points


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts

def data_augmentation_z(verts, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rng = np.random.RandomState()
    angle = rng.uniform(-180, 180) / 180.0 * np.pi   ## multiway 
    rot_matrix = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ], dtype=np.float32)
    rotation_matrix = torch.from_numpy(rot_matrix).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts




def augment_batch(data, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    data["shape1"]["xyz"] = data_augmentation(data["shape1"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)
    data["shape2"]["xyz"] = data_augmentation(data["shape2"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)

    return data


def data_augmentation_sym(shape):
    """
    we symmetrise the shape which results in conjugation of complex info
    """
    shape["gradY"] = -shape["gradY"]  # gradients get conjugated

    # so should complex data (to double check)
    shape["cevecs"] = torch.conj(shape["cevecs"])
    shape["spec_grad"] = torch.conj(shape["spec_grad"])
    if "vts_sym" in shape:
        shape["vts"] = shape["vts_sym"]


def augment_batch_sym(data, rand=True):
    """
    if rand = False : (test time with sym only) we symmetrize the shape
    if rand = True  : with a probability of 0.5 we symmetrize the shape
    """
    #print(data["shape1"]["gradY"][0,0])
    if not rand or random.randint(0, 1) == 1:
        # print("sym")
        data_augmentation_sym(data["shape1"])
    #print(data["shape1"]["gradY"][0,0], data["shape2"]["gradY"][0,0])
    return data


def auto_WKS(evals, evects, num_E, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
    sigma = 7*(e_max-e_min)/num_E

    e_min += 2*sigma
    e_max -= 2*sigma

    energy_list = np.linspace(e_min, e_max, num_E)

    return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :])/(2*sigma**2))  # (num_E,K)

    weighted_evects = evects[None, :, :] * coefs[:, None, :]  # (num_E,N,K)

    natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None, :] * natural_WKS

    else:
        return natural_WKS


def read_geodist(mat):
    # get geodist matrix
    if 'Gamma' in mat:
        G_s = mat['Gamma']
    elif 'G' in mat:
        G_s = mat['G']
    else:
        raise NotImplementedError('no geodist file found or not under name "G" or "Gamma"')

    # get square of mesh area
    if 'SQRarea' in mat:
        SQ_s = mat['SQRarea'][0]
        # print("from mat:", SQ_s)
    else:
        SQ_s = 1

    return G_s, SQ_s

import sys
import os
import time

import torch
import hashlib
import numpy as np
import scipy

# == Pytorch things

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

# Numpy things

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)




# This function and the helper class below are to support parallel computation of all-pairs geodesic distance
def all_pairs_geodesic_worker(verts, faces, i):
    import igl

    N = verts.shape[0]

    # TODO: this re-does a ton of work, since it is called independently each time. Some custom C++ code could surely make it faster.
    sources = np.array([i])[:,np.newaxis]
    targets = np.arange(N)[:,np.newaxis]
    dist_vec = igl.exact_geodesic(verts, faces, sources, targets)
    
    return dist_vec
        
class AllPairsGeodesicEngine(object):
    def __init__(self, verts, faces):
        self.verts = verts 
        self.faces = faces 
    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)
from multiprocessing import Pool

def get_all_pairs_geodesic_distance(verts_np, faces_np, geodesic_cache_dir=None):
    """
    Return a gigantic VxV dense matrix containing the all-pairs geodesic distance matrix. Internally caches, recomputing only if necessary.
    (numpy in, numpy out)
    """

    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError("Must have python libigl installed for all-pairs geodesics. `conda install -c conda-forge igl`")

    # Check the cache
    found = False 
    if geodesic_cache_dir is not None:
        ensure_dir_exists(geodesic_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                geodesic_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache_search += 1
                    continue

                # This entry matches! Return it.
                found = True
                result_dists = npzfile["dist"]
                break

            except FileNotFoundError:
                break

    if not found:
                
        print("Computing all-pairs geodesic distance (warning: SLOW!)")

        # Not found, compute from scratch
        # warning: slowwwwwww

        N = verts_np.shape[0]

        try:
            pool = Pool(None) # on 8 processors
            engine = AllPairsGeodesicEngine(verts_np, faces_np)
            outputs = pool.map(engine, range(N))
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)
        result_dists = np.nan_to_num(result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist)

        print("...finished computing all-pairs geodesic distance")

        # put it in the cache if possible
        if geodesic_cache_dir is not None:

            print("saving geodesic distances to cache: " + str(geodesic_cache_dir))

            # TODO we're potentially saving a double precision but only using a single
            # precision here; could save storage by always saving as floats
            np.savez(search_path,
                     verts=verts_np,
                     faces=faces_np,
                     dist=result_dists
                     )

    return result_dists


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", points)
    for idx in range(0, len(points.points)):
        vis.add_3d_label(points.points[idx], "{}".format(idx))

    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()
