import numpy as np
from PIL import Image
import cv2
import json
import os
import open3d as o3d
from nerfies.camera import Camera
import scipy.linalg as linalg
from tqdm import tqdm


W_idx = []
for i in range(16):
    W_idx.append(np.array([i,i]))
for j in range(16):
    W_idx.append(np.array([j,31-j]))
for i in range(16):
    W_idx.append(np.array([i,32+i]))
for j in range(16):
    W_idx.append(np.array([j,63-j]))


def create_eyes(eye, axis=np.array([-0., 1., 0]), number_of_views=16):
    eyes = [eye]
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * np.pi / (number_of_views / 2)))
    for i in range(number_of_views - 1):
        eye = np.dot(rot_matrix, eye)
        eyes.append(eye)
    return eyes

def create_eyes_sphere2(n_points, radius):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)
    return np.vstack((x, y, z)).T


def create_eyes_sphere(number_of_views=8, radius=3.5):
    phi = np.linspace(0, np.pi, number_of_views)
    theta = np.linspace(0, 2 * np.pi, number_of_views)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    

    return np.vstack((x.flatten(), y.flatten(), z.flatten())).T


def make_origin_camera(orientation = np.eye(3), focal_length = 955.40500674, position = np.array([0,0,3.5]), img_size = np.array([512, 512]), principal_point = np.array([256, 256])):
    

    cam_origin = Camera(orientation, position, focal_length, principal_point, img_size)
    return cam_origin


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def convert_npy_to_dataset(images, dataset_name, eyes, n_inter_frames=16, n_views=16, center = np.array([0.,0.,0.]),up = np.array([0.,1.,0.])):

    dataset_json = {
        "count": 0,
        "num_exemplars": 0
    }
    metadata_json = {}
    scene_json = {
    "scale": 1.0,
    "scene_to_metric": 1.0,
    "center": [
        0.,
        0.,
        0.
    ],
    "near": 0.008369162926060075,
    "far": 50
    }
    ids = []
    train_ids = []
    val_ids = []
    cameras = []
    cameras_train = []
    cameras_vrig = []

    count = 0
    num_exemplars = 0

    if not os.path.exists("./data/{}/rgb/1x".format(dataset_name)):
        os.makedirs("./data/{}/rgb/1x".format(dataset_name))
    if not os.path.exists("./data/{}/camera".format(dataset_name)):
        os.makedirs("./data/{}/camera".format(dataset_name))
    cam_origin = make_origin_camera()
    for i in range(n_inter_frames):
        for j in range(n_views):
            cv2.imwrite("./data/{}/rgb/1x/inter{}_view{}.png".format(dataset_name, i,j), images[i][j])
            for k in [2, 4, 8, 16]:
                if not os.path.exists("./data/{}/rgb/{}x/".format(dataset_name, k)):
                    os.makedirs("./data/{}/rgb/{}x/".format(dataset_name, k))
                downsampled_img = cv2.resize(images[i][j], (0, 0), fx=1. / k, fy=1. / k, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite("./data/{}/rgb/{}x/inter{}_view{}.png".format(dataset_name, k,i,j), downsampled_img)

            ids.append("inter{}_view{}".format(i,j))
            camera = cam_origin.look_at(eyes[j], center, up)
            cameras.append(camera)
            camera_json = camera.to_json()
            camera_json['image_size'] = [512,512]
            save_json(camera_json, "./data/{}/camera/inter{}_view{}.json".format(dataset_name, i,j))
            if False:

                val_ids.append("inter{}_view{}".format(i,j))
                cameras_vrig.append(camera)
            else:
                train_ids.append("inter{}_view{}".format(i,j))
                cameras_train.append(camera)
                num_exemplars += 1
            
            metadata_json["inter{}_view{}".format(i,j)] = {
                "time_id": i,
                "warp_id": i,
                "appearance_id": i,
                "camera_id": i
            }
            count += 1

    dataset_json["count"] = count
    dataset_json["num_exemplars"] = num_exemplars
    dataset_json['ids'] = ids
    dataset_json['train_ids'] = train_ids
    dataset_json['val_ids'] = val_ids

    save_json(dataset_json, "./data/{}/dataset.json".format(dataset_name))
    save_json(metadata_json, "./data/{}/metadata.json".format(dataset_name))
    save_json(scene_json, "./data/{}/scene.json".format(dataset_name))


if __name__ == "__main__":
    for file in tqdm(os.listdir("./data/")):
        if not file.endswith(".npy"):
            continue
        if file.endswith(".npy") and file.split('.')[0] in os.listdir("./data/"):
            continue
        if file.split('.')[0] == "71_63":
            continue
        print(file)
        dataset_name = file.split('.')[0]
        if not os.path.exists("./data/{}".format(dataset_name)):
            os.makedirs("./data/{}".format(dataset_name))
        images = np.load("./data/{}.npy".format(dataset_name))[:,:,:,:,::-1]
        n_views = 32
        eyes = create_eyes_sphere2(n_views, 3.5)
        selected_views = [0, 30, 21, 15, 9, 19, 18, 12, 8, 6, 22, 28, 29, 5, 7, 16]
        selected_views.sort()
        eyes = eyes[selected_views]
        mesh = o3d.io.read_triangle_mesh("./data/meshes/{}.off".format(file.split('.')[0][0:10]))
        np.save("./data/{}/points.npy".format(dataset_name), np.asarray(mesh.vertices) )
        convert_npy_to_dataset(images, dataset_name, eyes, n_inter_frames=10)