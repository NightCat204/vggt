import json
import math
import os
import numpy as np
import cv2
import torch
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.renderer import FoVPerspectiveCameras
import viser
from pathlib import Path


def map_paths(path: str, input_type: str, output_type: str) -> str:
    mapping = {
        "rgb": {
            "depth": ("/rgb/", "/depth_zbuffer/", "_rgb.png", "_depth_zbuffer.png"),
            "npz":   ("/rgb/", "/camera_pose/",   "_rgb.png", "_fixatedpose.npz"),
        },
        "depth": {
            "rgb":  ("/depth_zbuffer/", "/rgb/",           "_depth_zbuffer.png", "_rgb.png"),
            "npz":  ("/depth_zbuffer/", "/camera_pose/",   "_depth_zbuffer.png", "_fixatedpose.npz"),
        },
        "npz": {
            "rgb":  ("/camera_pose/",   "/rgb/",           "_fixatedpose.npz", "_rgb.png"),
            "depth":("/camera_pose/",   "/depth_zbuffer/", "_fixatedpose.npz", "_depth_zbuffer.png"),
        },
        "npz_cam": {
            "rgb":  ("/camera_pose/",   "/rgb/",           "_camera_pose.npz", "_rgb.png"),
            "depth":("/camera_pose/",   "/depth_zbuffer/", "_camera_pose.npz", "_depth_zbuffer.png"),
        }
    }

    if input_type == output_type:
        return path  

    try:
        in_dir, out_dir, in_suffix, out_suffix = mapping[input_type][output_type]
    except KeyError:
        raise ValueError(f"Invalid input_type/output_type: {input_type} -> {output_type}")

    return path.replace(in_dir, out_dir).replace(in_suffix, out_suffix)

def camera_pose_dir_from_sim(sim_path: str) -> str:
    """
    e.g.
    sim_path = /.../omnidata_starter_dataset/similarity/replica/apartment_0/
    ->        /.../omnidata_starter_dataset/camera_pose/replica/apartment_0
    """
    sim_path = sim_path.rstrip("/ ")
    if "/similarity/" not in sim_path:
        raise ValueError(f"sim_path missing '/similarity/': {sim_path}")
    return sim_path.replace("/similarity/", "/camera_pose/").rstrip("/")


def _get_cam_to_world_R_T_K(point_info):
    EULER_X_OFFSET_RADS = math.radians(90.0)
    location = point_info['camera_location']
    rotation = point_info['camera_rotation_final']
    fov      = point_info['field_of_view_rads']

    # Recover cam -> world
    ex, ey, ez = rotation
    R     = euler_angles_to_matrix(torch.tensor(
                [(ex - EULER_X_OFFSET_RADS, -ey, -ez)],
                dtype=torch.double), 'XZY')
    Tx, Ty, Tz = location
    T     = torch.tensor([[-Tx, Tz, Ty]], dtype=torch.double) 

    # P3D expects world -> cam
    R_inv = R.transpose(1,2)
    T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    # T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    # T_inv = T
    # R_inv = R 
    K = FoVPerspectiveCameras(R=R_inv, T=T_inv, fov=fov, degrees=False).compute_projection_matrix(znear=0.001, zfar=512.0, fov=fov, aspect_ratio=1.0, degrees=False)
    
    return dict(
        cam_to_world_R=R_inv.squeeze(0).float(),
        cam_to_world_T=T_inv.squeeze(0).float(),
        proj_K=K.squeeze(0).float(),
        proj_K_inv=K[:,:3,:3].inverse().squeeze(0).float())


def project_rgb_to_point_cloud(rgb_image, depth_map, K, R, t):
    """
    unproject RGB-D image to point cloud.
    camera coordinate system: x-right, y-down, z-forward.
    3D coordinates in world coordinate system can project to 2D homogeneous coordinates by: x' = K * [R|t] * X

    Parameters:
    - rgb_image: RGB image
    - depth_map: metric depth map
    - K: camera intrinsic matrix, i.e. [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    - R: camera rotation matrix, i.e. world to camera
    - t: camera translation vector, i.e. world to camera

    Returns:
    - point_cloud: point cloud with color, Nx6 array, each row is [x, y, z, r, g, b]
    """
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # construct homogeneous pixel coordinates
    uv1 = np.stack([i.ravel() + 0.5, j.ravel() + 0.5, np.ones_like(i).ravel()], axis=1)
    
    # pixel coordinates -> camera coordinates
    K_inv = np.linalg.inv(K)
    xyz_camera = K_inv @ (uv1.T * depth_map.ravel())

    # camera coordinates -> world coordinates
    R_inv = R.T # camera to world
    t_inv = -R_inv @ t # camera to world
    xyz_world = R_inv @ xyz_camera + t_inv[:, None]
    xyz_world = xyz_world.T

    # point cloud with color
    colors = rgb_image.reshape(-1, 3)
    point_cloud = np.hstack([xyz_world, colors])

    # filter out invalid points
    valid_mask = (depth_map.ravel() > 0) & (depth_map.ravel() < 100)
    point_cloud = point_cloud[valid_mask]

    return point_cloud

def save_point_cloud(points_3d, save_path, binary=True):
    """
    Save point cloud to disk

    Parameters:
    - points_3d: point cloud with color, Nx6 array, each row is [x, y, z, r, g, b]
    - save_path: path to save the point cloud
    """
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(save_path)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(save_path, np.column_stack([x, y, z, r, g, b]), fmt='%f %f %f %d %d %d', header=ply_head, comments='')


def load_camera_from_npz(npz_path: str):
    data = np.load(npz_path)
    K = data["K"].astype(np.float64)              # (3,3)
    R = data["R"].astype(np.float64)              # (3,3) world->camera
    t = data["t"].astype(np.float64).reshape(3,)  # (3,)
    return K, R, t


def unproject_omni(rgb_path, pc_path=None, save_binary=True):

    depth_path = map_paths(rgb_path, "rgb", "depth")
    point_info_path = map_paths(rgb_path, "rgb", "npz")

    rgb_image = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 512.0

    # camera intrinsics and pose
    with open(point_info_path, 'r') as f:
        omnidata = json.load(f)
    # load R, T, K, the function returns what is actually world_to_camera
    rtk = _get_cam_to_world_R_T_K(omnidata)

    # camera intrinsics, convert to OpenCV format
    P = rtk['proj_K'].numpy() # OpenGL projection matrix
    w = h = 512 # omnidata should all be 512
    fx, fy, cx, cy = P[0,0]*w/2, P[1,1]*h/2, (w-P[0,2]*w)/2, (P[1,2]*h+h)/2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # camera extrinsics/pose, convert to OpenCV format
    world_to_cam_R_r = rtk['cam_to_world_R'].numpy() # for right multiply
    world_to_cam_R = world_to_cam_R_r.T # for left multiply
    world_to_cam_T = rtk['cam_to_world_T'].numpy()
    # Coordinate system transformation, i.e. pose in pytorch3d -> pose in OpenCV
    # Pytorch3D: right-handed, x-left, y-up, z-forward
    # OpenCV: right-handed, x-right, y-down, z-forward
    pytorch3d_to_opencv = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    world_to_cam_R = pytorch3d_to_opencv @ world_to_cam_R
    world_to_cam_T = pytorch3d_to_opencv @ world_to_cam_T

    # Now we get the R, T, K in OpenCV format
    pc = project_rgb_to_point_cloud(rgb_image, depth_map, K, world_to_cam_R, world_to_cam_T)
    if pc_path is not None:
        save_point_cloud(pc, pc_path, binary=save_binary)
    return pc

def visualize_with_viser(point_clouds, names=None, point_size=0.01, port=8080):
    if isinstance(point_clouds, np.ndarray):
        point_clouds = [point_clouds]
    if names is None:
        names = [f"/pc_{i}" for i in range(len(point_clouds))]

    server = viser.ViserServer(port=port)
    server.scene.add_frame(name="/world")

    for pc, name in zip(point_clouds, names):
        pts = pc[:, :3].astype(np.float32)
        cols = pc[:, 3:6].astype(np.uint8)
        server.scene.add_point_cloud(
            name=name,
            points=pts,
            colors=cols,
            point_size=point_size,
        )
    print(f"Viser started: http://localhost:{port}")
    import time
    while True:
        time.sleep(1)



def unproject_one(rgb_path: str, depth_path: str, npz_path: str):
    if not Path(rgb_path).exists():   raise FileNotFoundError(rgb_path)
    if not Path(depth_path).exists(): raise FileNotFoundError(depth_path)
    if not Path(npz_path).exists():   raise FileNotFoundError(npz_path)

    rgb  = cv2.imread(rgb_path)[:, :, ::-1]  # BGR->RGB
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_raw.astype(np.float32) / 512.0

    K, R, t = load_camera_from_npz(npz_path)
    pc = project_rgb_to_point_cloud(rgb, depth, K, R, t)
    return pc

def viser_from_similarity_json(sim_json: str, start: int = 0, end: int = 1,
                               point_size: float = 0.01, port: int = 8080,
                               max_frames: int | None = None):
    with open(sim_json, "r") as f:
        sim = json.load(f)
    ordered = sim["ordered"]           # from most similar (index 0) to least
    total = len(ordered)

    start = max(0, min(start, total))
    end   = max(start+1, min(end, total))

    # Always add first frame
    subset = [ordered[0]] + ordered[start:end]

    if max_frames is not None:
        subset = subset[:max_frames]

    pcs, names = [], []
    for rec in subset:
        pc = unproject_one(rec["rgb"], rec["depth"], rec["npz"])
        pcs.append(pc)
        # name contains index and distance, for easy recognition in viser sidebar
        nm = f"/{Path(rec['npz']).stem}_d{rec['distance']:.4f}"
        names.append(nm)

    visualize_with_viser(pcs, names=names, point_size=point_size, port=port)

def rotmat_to_wxyz(R_cw: np.ndarray) -> tuple[float, float, float, float]:
    m = R_cw.astype(np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return (float(w), float(x), float(y), float(z))

def fov_aspect_from_K(K: np.ndarray, img_w: int, img_h: int) -> tuple[float, float]:
    fy = float(K[1,1])
    fov_y = 2.0 * np.arctan( (img_h * 0.5) / fy )
    aspect = float(img_w) / float(img_h)
    return fov_y, aspect
