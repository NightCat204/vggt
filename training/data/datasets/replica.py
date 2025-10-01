import os
import os.path as osp
import glob
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from data.utils import map_paths, load_camera_from_npz
from data.dataset_util import *
from data.base_dataset import BaseDataset

from hydra import initialize, compose



class ReplicaDataset(BaseDataset):
    """
    Root Path Example:
        CAMERA_POSE_ROOT = /lustre/.../Omnidata/omnidata_starter_dataset/camera_pose/replica/
    Sequence Structure:
        camera_pose/replica/<scene>/*.npz
        rgb/replica/<scene>/*.png
        depth_zbuffer/replica/<scene>/*.png
        similarity/replica/<scene>/dists.npz
    """
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CAMERA_POSE_ROOT: str = "/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Omnidata/omnidata_starter_dataset/camera_pose/replica/",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.CAMERA_POSE_ROOT = CAMERA_POSE_ROOT.rstrip("/")
        self.min_num_images = min_num_images


        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
            
        logging.info(f"Replica CAMERA_POSE_ROOT = {self.CAMERA_POSE_ROOT}")

        scene_dirs = [p for p in glob.glob(osp.join(self.CAMERA_POSE_ROOT, "*")) if osp.isdir(p)]
        scene_dirs = sorted(scene_dirs)
        sequence_list = []
        seq_lengths = []
        pose_list_dict = {}
        dists_npz_dict = {}

        for scene_dir in scene_dirs:
            pose_list = sorted(glob.glob(osp.join(scene_dir, "*.npz")))
            if len(pose_list) >= self.min_num_images:
                sim_dir = scene_dir.replace("/camera_pose/", "/similarity/").rstrip("/")
                dists_npz = osp.join(sim_dir, "dists.npz")
                if not osp.exists(dists_npz):
                    logging.warning(f"No dists.npz in {sim_dir}")
                    continue
                sequence_list.append(scene_dir)
                seq_lengths.append(len(pose_list))
                pose_list_dict[scene_dir] = pose_list
                dists_npz_dict[scene_dir] = np.load(dists_npz)

        self.dists_npz_dict = dists_npz_dict
        self.pose_list_dict = pose_list_dict
        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)
        self.save_seq_stats(self.sequence_list, seq_lengths, self.__class__.__name__) 

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Replica scene count: {self.sequence_list_len}")
        logging.info(f"{status}: Replica dataset length: {len(self)}")


    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,   
        ids: list = None,       
        aspect_ratio: float = 1.0,
    ) -> dict:

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        scene_dir = self.sequence_list[seq_index]  

        pose_list = self.pose_list_dict[scene_dir]
        num_images = len(pose_list)
        if num_images < 1:
            raise RuntimeError(f"No npz in scene_dir={scene_dir}")

        sim_mat = self.dists_npz_dict[scene_dir]
        ranking = sim_mat["ranking"]  # (N,N)
        # dists   = sim_mat["dists"]    # (N,N)
        assert ranking.shape[0] == num_images, f"ranking rows != num_images ({ranking.shape[0]} vs {num_images})"


        anchor_idx = random.randint(0, num_images - 1)
        # print(f"anchor_idx = {anchor_idx}")
        order = ranking[anchor_idx].tolist()
        order = [i for i in order if i != anchor_idx]
        topK  = order[:min(127, len(order))]
        pick = sorted(random.sample(topK, img_per_seq - 1))
        ids = [anchor_idx] + pick  


        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points = [], []
        point_masks, original_sizes = [], []

        for local_idx in ids:
            npz_path = pose_list[local_idx]
            rgb_path = map_paths(npz_path, "npz_cam", "rgb")
            depth_path = map_paths(npz_path, "npz_cam", "depth")

            # print(f"rgb_path = {rgb_path}")
            # print(f"depth_path = {depth_path}")

            image = read_image_cv2(rgb_path) 
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 512.0
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max) 

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            K, R, t = load_camera_from_npz(npz_path)

            extri_opencv = np.zeros((3, 4), dtype=np.float32)
            extri_opencv[:3, :3] = R
            extri_opencv[:3, 3] = t
            intri_opencv = K.copy()

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=rgb_path,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {scene_dir}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "replica"
        batch = {
            "seq_name": set_name + "_" + Path(scene_dir).name,
            "ids": np.array(ids),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,   
            "intrinsics": intrinsics,   
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch

if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    dataset = ReplicaDataset(common_conf=cfg.data.train.common_config, split="train", CAMERA_POSE_ROOT="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Omnidata/omnidata_starter_dataset/camera_pose/replica/")
    dataset[(0, 128, 1.0)]