import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import cv2
import viser
from pathlib import Path
from hydra import initialize, compose
from data.datasets.replica import ReplicaDataset
from data.datasets.blendedmvs import BlendedMVSDataset
from data.datasets.vkitti import VKittiDataset
from data.datasets.wildrgbd import WildRGBDDataset
from data.datasets.MVSSynth import MVSSynthDataset
from data.datasets.ASE import ASEDataset
from data.datasets.ADT import ADTDataset
from data.datasets.mapillary import MapillaryDataset
from data.datasets.megadepth import MegaDepthDataset
from data.datasets.hypersim import HyperSimDataset


def rotmat_to_wxyz(R_cw: np.ndarray) -> tuple[float, float, float, float]:
    m = R_cw.astype(np.float64)
    tr = np.trace(m)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
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
    fov_y = 2.0 * np.arctan((img_h * 0.5) / fy) 
    aspect = float(img_w) / float(img_h)
    return fov_y, aspect

def project_rgbd_to_world(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    uv1 = np.stack([i.ravel() + 0.5, j.ravel() + 0.5, np.ones_like(i).ravel()], axis=0)  # (3,N)

    z = depth.ravel().astype(np.float32)
    valid = (z > 0) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((0,6), dtype=np.float32)

    K_inv = np.linalg.inv(K).astype(np.float32)
    xyz_cam = (K_inv @ (uv1 * z))  # (3,N)

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc.reshape(3,)
    xyz_w = (R_cw @ xyz_cam + t_cw[:,None]).T  # (N,3)

    rgb_flat = rgb.reshape(-1,3).astype(np.float32)
    xyz_w = xyz_w[valid]
    rgb_sel = rgb_flat[valid]

    pc = np.hstack([xyz_w, rgb_sel]).astype(np.float32)
    return pc

def visualize_batch_with_viser(batch: dict, point_size: float = 0.01, port: int = 8080,
                               show_frustum: bool = True, frustum_scale: float = 0.25):
    server = viser.ViserServer(port=port)
    url = server.request_share_url()
    print("Public share URL:", url)

    server.scene.add_frame(name="/world")

    images       = batch["images"]
    depths       = batch["depths"]
    extrinsics   = batch["extrinsics"]  # list of (3,4) [R|t] world->cam
    intrinsics   = batch["intrinsics"]  # list of (3,3) K
    frame_num    = batch["frame_num"]
    seq_name     = batch["seq_name"]

    print(f"[Viser] seq={seq_name}, frames={frame_num}")

    # Add depth analysis
    depth_max = 0
    depth_min = 0
    depth_mean = 0

    for k in range(frame_num):
        img = images[k]                      
        dep = depths[k].astype(np.float32)   
        E   = extrinsics[k].astype(np.float32)
        K   = intrinsics[k].astype(np.float32)

        R = E[:, :3]
        t = E[:,  3]

        depth_max = max(depth_max, dep.max())
        depth_min = min(depth_min, dep.min())
        depth_mean = depth_mean + dep.mean()

        pc = project_rgbd_to_world(img, dep, K, R, t)
        name = f"/frame_{k:04d}"
        if pc.shape[0] > 0:
            server.scene.add_point_cloud(
                name=name,
                points=pc[:, :3],
                colors=pc[:, 3:6].astype(np.uint8),
                point_size=point_size,
            )

        if show_frustum:
            H, W = img.shape[:2]
            R_cw = R.T
            C = -R_cw @ t
            wxyz = rotmat_to_wxyz(R_cw)
            fov_y, aspect = fov_aspect_from_K(K, W, H)
            server.scene.add_camera_frustum(
                name=name + "_frustum",
                fov=float(fov_y),
                aspect=float(aspect),
                wxyz=wxyz,
                position=tuple(C.tolist()),
                scale=frustum_scale,
                color=(78,120,192) if k == 0 else (60,60,60),
                image=img,  
            )

    depth_mean = depth_mean / frame_num
    server.gui.add_markdown(
        content=f"Depth max: {depth_max}, min: {depth_min}, mean: {depth_mean}",
    )
    print(f"Depth max: {depth_max}, min: {depth_min}, mean: {depth_mean}")

    import time
    while True:
        time.sleep(1)

# Add PATH here
PATH_ROOT_DICT = {
    'replica': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Omnidata/omnidata_starter_dataset/camera_pose/replica/',
    'blendedmvs': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/BlendedMVS/BlendedMVS',
    'vkitti': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/vkitti',
    'wildrgbd': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/WildRGB-D',
    'MVSSynth': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/MVS-Synth/GTAV_720',
    'ASE': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ASE',
    'ADT': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ADT',
    'mapillary': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Mapillary',
    'megadepth': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/MegaDepth/phoenix/S6/zl548/MegaDepth_v1',
    'hypersim': '/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Omnidata/omnidata_starter_dataset/camera_pose/hypersim/',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_index", type=int, default=10)
    parser.add_argument("--img_per_seq", type=int, default=24)
    parser.add_argument("--point_size", type=float, default=0.005)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--show_frustum", action="store_true", default=True)
    parser.add_argument("--frustum_scale", type=float, default=0.25)
    parser.add_argument("--dataset", type=str, default="hypersim")  # Change to your dataset
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default")

    if args.dataset == "replica":
        ds = ReplicaDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            CAMERA_POSE_ROOT=PATH_ROOT_DICT[args.dataset],
            min_num_images=24,
            len_train=100000,
            len_test=10000,
        )
    elif args.dataset == "blendedmvs":
        ds = BlendedMVSDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            BLENDED_DIR=PATH_ROOT_DICT[args.dataset],
            min_num_images=24,
            len_train=100000,
            len_test=10000,
        )
    elif args.dataset == "vkitti":
        ds = VKittiDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            VKitti_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "wildrgbd":
        ds = WildRGBDDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            WILDRGBD_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "MVSSynth":
        ds = MVSSynthDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            MVSSynth_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "ASE":
        ds = ASEDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            ASE_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "ADT":
        ds = ADTDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            ADT_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "mapillary":
        ds = MapillaryDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            MAPILLARY_DIR=PATH_ROOT_DICT[args.dataset],
        )
    elif args.dataset == "megadepth":
        ds = MegaDepthDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            MEGADEPTH_DIR=PATH_ROOT_DICT[args.dataset],
        )   
    elif args.dataset == "hypersim":
        ds = HyperSimDataset(
            common_conf=cfg.data.train.common_config,
            split="train",
            CAMERA_POSE_ROOT=PATH_ROOT_DICT[args.dataset],
        )   
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    batch = ds.get_data(seq_index=args.seq_index, img_per_seq=args.img_per_seq, aspect_ratio=1.0)

    visualize_batch_with_viser(
        batch,
        point_size=args.point_size,
        port=args.port,
        show_frustum=args.show_frustum,
        frustum_scale=args.frustum_scale,
    )

if __name__ == "__main__":
    main()
