import numpy as np
import open3d as o3d
import cv2
import copy
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import os
import shutil
from itertools import permutations
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm
from itertools import combinations
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def rotation_matrix_y(theta):
    theta = np.radians(theta)  
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    return rotation_matrix


def create_eyes(eye, axis=np.array([-0., 1., 0]), number_of_views=16):
    eyes = [eye]
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * np.pi / (number_of_views / 2)))
    for i in range(number_of_views - 1):
        eye = np.dot(rot_matrix, eye)
        eyes.append(eye)
    return eyes


def create_eyes_6(radius=3.5):
    return [
        [0, 0, radius],
        [0, 0, -radius],
        [radius, 0, 0],
        [-radius, 0, 0],
        [0, radius, 0],
        [0, -radius, 0]
    ]

def cal_center_on_sphere(p1, p2, center=np.array([0,0,0]), radius=3.5):
    temp = np.array(p1) + np.array(p2)
    if (temp == np.array([0,0,0])).all():
        return np.array([0,0,0])
    k = np.abs(np.sqrt(radius ** 2 / (temp[0] ** 2 + temp[1] ** 2 + temp[2] ** 2)))

    return k * temp


def create_eyes_21(radius=3.5):
    eyes_6 = create_eyes_6()
    eyes_15 = []
    for combination in combinations(eyes_6, 2):
        p1 = combination[0]
        p2 = combination[1]
        eye = cal_center_on_sphere(p1, p2)
        if (eye == np.array([0,0,0])).all():
            continue
        eyes_15.append(eye)
    return eyes_15 + eyes_6


def create_eyes_sphere(n_points, radius):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)
    return np.vstack((x, y, z)).T



def create_render(bg_image=None, img_width=800, img_height=800):
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    render.scene.set_background([0.,0.0,0.,1.], bg_image)  # RGBA
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, [0,0,0])
    aspect_ratio = img_width / img_height
    return render

def render_mesh(mesh, render, center, eye, up, mesh_name='mesh'):
    render.scene.remove_geometry(mesh_name)
    mtl = o3d.visualization.rendering.MaterialRecord() 
    mtl.base_color = [0.5, 0.5, 0.5, 0.9] 
    mtl.shader = "defaultLit"
    render.setup_camera(30.0, center, eye, up)

    render.scene.add_geometry(mesh_name, mesh, mtl)
    img_o3d = render.render_to_image()
    img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_BGR2RGB)
    render.scene.remove_geometry(mesh_name)
    return img_cv2

def pcd_depth_colormap(pcd):
    points = np.asarray(pcd.points)
    num_points = len(points)
    depth_values = points[:, 2]
    normalized_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))

    color_values = (normalized_depth * 65535).astype(int)

    colors = np.zeros((num_points, 3))
    colors[:, 0] = (color_values / 255).astype(int)  
    colors[:, 1] = 127
    colors[:, 2] = color_values - (color_values / 255).astype(int) * 255 
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
def colorize_vertices_by_depth(mesh: o3d.geometry.TriangleMesh, colormap=plt.cm.Set3_r):
    vertices = np.asarray(mesh.vertices)
    depth = vertices[:, 2]
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    color_values = (normalized_depth * 65535).astype(int)
    colors = np.zeros((vertices.shape[0], 3))
    colors[:, 0] = (color_values / 255).astype(int) 
    colors[:, 1] = (color_values / 255).astype(int)  
    colors[:, 2] = color_values - (color_values / 255).astype(int) * 255  
    colors /= 255.
    return colors

def colorize_vertices_by_normal(mesh):
    mesh.compute_vertex_normals()

    normals = np.asarray(mesh.vertex_normals)

    colors = []
    for normal in normals:
        color = (normal + 1) / 2 
        colors.append(color)

    return np.array(colors)


def colorize_vertices_by_y(mesh: o3d.geometry.TriangleMesh, colormap=plt.cm.Set3_r):
    vertices = np.asarray(mesh.vertices)
    depth = vertices[:, 1]
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    colors = colormap(normalized_depth)[:, :3]
    return colors

def colorize_vertices_by_x(mesh: o3d.geometry.TriangleMesh, colormap=plt.cm.Set3_r):
    vertices = np.asarray(mesh.vertices)
    depth = vertices[:, 0]
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    colors = colormap(normalized_depth)[:, :3]
    return colors

def colorize_points_by_depth(pcd: o3d.geometry.PointCloud, colormap=plt.cm.Set3_r):
    points = np.asarray(pcd.points)
    depth = points[:, 2]
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    color_values = (normalized_depth * 65535).astype(int)
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = (color_values / 255).astype(int) 
    colors[:, 1] = (color_values / 255).astype(int)  
    colors[:, 2] = color_values - (color_values / 255).astype(int) * 255  
    colors /= 255.
    return colors
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

def render_folder(folder_path, save_image_path, render, eyes, convert_to_pointcloud=False, faces_color=False):
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    for mesh_path in tqdm(os.listdir(folder_path)):
        if len(mesh_path.split('.')) <= 1:
            continue
        mesh_folder_path = os.path.join(folder_path, mesh_path)
        mesh_name = ''.join(mesh_path).split('.')[0]
        to_render = [
            "103_rot",
            "387_rot"
        ]
        if mesh_name in to_render:
            continue

        mesh = o3d.io.read_triangle_mesh(mesh_folder_path)
        mesh.compute_vertex_normals()
        color_z_1 = colorize_vertices_by_depth(mesh, plt.cm.autumn)
        color = color_z_1
        mesh.vertex_colors = o3d.utility.Vector3dVector((color))

        if convert_to_pointcloud:
            pcd = o3d.geometry.PointCloud()
            points = np.asarray(mesh.vertices)
            
            pcd.points = o3d.utility.Vector3dVector(points)
            
            pcd.colors = o3d.utility.Vector3dVector(colorize_points_by_depth(pcd))
            mesh = pcd
            o3d.io.write_point_cloud(os.path.join(folder_path, "../colored/" + mesh_name + "colored.ply"), mesh)
            np.save(os.path.join(folder_path, "../colored/" + mesh_name + "colored.npy"), points)
        if faces_color and not convert_to_pointcloud:
            pass
        img_cv2 = render_mesh(mesh=mesh, render=render, center=[0, 0, 0], up=[0., 1, 0], eye=eyes[0])
        if not convert_to_pointcloud:
            o3d.io.write_triangle_mesh(os.path.join(folder_path, "../colored/" + mesh_name + "colored.ply"), mesh)
        for i in range(len(eyes)):
            light_position = eyes[i].copy()
            light_position += np.array([0 , 6, 0])
            img_cv2 = render_mesh(mesh=mesh, render=render, center=[0, 0, 0], up=[0., 1, 0], eye=eyes[i])
            img_output_path = os.path.join(save_image_path, mesh_name + "_view" + str(i) + ".png")
            cv2.imwrite(img_output_path, img_cv2)
    cv2.destroyAllWindows()
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def mesh_normalize(mesh):
    v = np.asarray(mesh.vertices)
    v_normalized = pc_normalize(v)
    mesh.vertices = o3d.utility.Vector3dVector(v_normalized)
    return mesh

if __name__ == "__main__":
    mesh_folder_path = "./meshes/cross"
    save_image_path = "./rendered_images/"
    render = create_render()

    num_views = 32
    eyes = create_eyes_sphere(num_views, 3.5)
    addtional = np.array([
        [0, 0, 3],
        [0, 1, 3],
        [0, 2, 2],
        [0, 1.5, 2.5]
    ])
    render_folder(mesh_folder_path, save_image_path, render=render, eyes=eyes, convert_to_pointcloud=False, faces_color=False)

