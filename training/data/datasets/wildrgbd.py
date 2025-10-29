# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob
import json

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset

from hydra import initialize, compose


class WildRGBDDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        WILDRGBD_DIR: str = "/path/to/wildrgbd",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8
    ):
        """
        Initialize the WildRGBDDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            WILDRGBD_DIR (str): Root directory path to WildRGB-D data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_ratio (int): Range for expanding nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.WILDRGBD_DIR = WILDRGBD_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"WILDRGBD_DIR is {self.WILDRGBD_DIR}")

        categories = sorted([
            d for d in os.listdir(self.WILDRGBD_DIR)
            if osp.isdir(osp.join(self.WILDRGBD_DIR, d, 'scenes'))
        ])

        self.sequence_list = []
        self._seq_len_cache = []
        for category in categories:
            category_dir = osp.join(self.WILDRGBD_DIR, category, 'scenes')
            seq_names = glob.glob(osp.join(category_dir, '*'))
            seq_names = sorted(list(seq_names))

            for seq in seq_names:
                scene_dir = osp.join(category_dir, seq)
                if not osp.isdir(scene_dir):
                    continue
                rgb_dir = osp.join(scene_dir, 'rgb')
                mask_dir = osp.join(scene_dir, 'masks')
                depth_dir = osp.join(scene_dir, 'depth')
                meta_path = osp.join(scene_dir, 'metadata')
                c2w_path = osp.join(scene_dir, 'cam_poses.txt')
                if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) \
                    and osp.isfile(meta_path) and osp.isfile(c2w_path)):
                    continue

                try:
                    c2w_content = np.genfromtxt(c2w_path)
                    if c2w_content.ndim == 1:
                        c2w_content = c2w_content[None, :]
                    frame_idx = c2w_content[:, 0].astype(np.int64)
                    num_frames = frame_idx.shape[0]
                except Exception as e:
                    logging.warning(f"Failed to read cam_poses for {scene_dir}: {e}")
                    continue

                if num_frames >= self.min_num_images:
                    self.sequence_list.append((category, seq))
                    self._seq_len_cache.append(num_frames)

        self.sequence_list_len = len(self.sequence_list)

        self.save_seq_stats([f"{c}/{s}" for c, s in self.sequence_list], self._seq_len_cache, self.__class__.__name__)

        self.depth_max = 80  # optional clamp (meters). Set -1 to disable in threshold_depth_map.

        status = "Training" if self.training else "Testing"
        print(f"{status}: WildRGB-D scene count: {self.sequence_list_len}")
        logging.info(f"{status}: WildRGB-D dataset length: {len(self)}")


    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,   
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): "category/sequence" name.
            ids (list): Specific IDs (frame indices) to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            category, scene = self.sequence_list[seq_index]
        else:
            if '/' in seq_name:
                category, scene = seq_name.split('/', 1)
            else:
                raise ValueError(f"Invalid sequence name: {seq_name}")

        scene_dir = osp.join(self.WILDRGBD_DIR, category, scene)
        rgb_dir = osp.join(scene_dir, 'rgb')
        depth_dir = osp.join(scene_dir, 'depth')
        mask_dir = osp.join(scene_dir, 'masks')
        meta_path = osp.join(scene_dir, 'metadata')
        cam2world_path = osp.join(scene_dir, 'cam_poses.txt')

        cam2world_content = np.genfromtxt(cam2world_path)
        if cam2world_content.ndim == 1:
            cam2world_content = cam2world_content[None, :]
        frame_idx = cam2world_content[:, 0].astype(np.int64)
        num_images = frame_idx.shape[0]

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []


        with open(meta_path, 'r') as f:
            meta = json.load(f)
        K_full = np.array(meta["K"]).reshape(3, 3).T
        fx, fy, cx, cy = K_full[0, 0], K_full[1, 1], K_full[0, 2], K_full[1, 2]
        intri_template = np.array([[fx, 0,  cx],
                                   [0,  fy, cy],
                                   [0,  0,  1]], dtype=np.float32)

        cam2world_all = cam2world_content[:, 1:].reshape(-1, 4, 4).astype(np.float32)
        assert cam2world_all.shape[0] == num_images, f"cam_poses size mismatch at {scene_dir}"

        for id in ids:
            stem5 = f"{id:05d}"
            image_filepath = osp.join(rgb_dir,   f"{stem5}.png")
            depth_filepath = osp.join(depth_dir, f"{stem5}.png")
            mask_filepath = osp.join(mask_dir, f"{stem5}.png")
            image = read_image_cv2(image_filepath)
            mask =  cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE) > 128

            # convert mm to m
            depth_raw = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
            depth_raw[~mask] = 0
            depth_map = threshold_depth_map(depth_raw, max_percentile=98, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, \
                f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2], dtype=np.int32)

            T_cw = cam2world_all[id]
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw

            extri_opencv = np.concatenate([R_wc, t_wc.reshape(3, 1)], axis=1).astype(np.float32)
            intri_opencv = intri_template.copy()

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
                filepath=image_filepath,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {category}/{scene}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "wildrgbd"
        batch = {
            "seq_name": set_name + "_" + f"{category}_{scene}",
            "ids": ids,
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


# In training folder, run: python data/datasets/wildrgbd.py
if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    dataset = WildRGBDDataset(
        common_conf=cfg.data.train.common_config,
        split="train",
        WILDRGBD_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/WildRGB-D"
    )
    print(dataset[(0, 24, 1.0)])
