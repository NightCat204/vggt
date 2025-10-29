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

import cv2
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset

from hydra import initialize, compose


class BlendedMVSDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        BLENDED_DIR: str = "/path/to/BlendedMVS",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the BlendedMVSDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            BLENDED_DIR (str): Directory path to BlendedMVS data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.BLENDED_DIR = BLENDED_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"BLENDED_DIR is {self.BLENDED_DIR}")
      
        sequence_list = [d for d in os.listdir(self.BLENDED_DIR) if osp.isdir(osp.join(self.BLENDED_DIR, d)) and len(d) == 24]
        sequence_list = sorted(sequence_list)

        # Filter sequences with enough images available
        filtered_list = []
        for seq in sequence_list:
            img_dir = osp.join(self.BLENDED_DIR, seq, 'blended_images')
            cam_dir = osp.join(self.BLENDED_DIR, seq, 'cams')
            depth_dir = osp.join(self.BLENDED_DIR, seq, 'rendered_depth_maps')
            if not (osp.isdir(img_dir) and osp.isdir(cam_dir) and osp.isdir(depth_dir)):
                continue
            cam_files = sorted([p for p in glob.glob(osp.join(cam_dir, "*_cam.txt")) if osp.basename(p) != 'pair.txt'])
            if len(cam_files) >= self.min_num_images:
                filtered_list.append(seq)

        self.sequence_list = filtered_list
        self.sequence_list_len = len(self.sequence_list)

        # Analyze the sequence number and the distribution of the sequence length
        sequence_lengths = [len(cam_files) for cam_files in self.sequence_list]
        self.save_seq_stats(self.sequence_list, sequence_lengths, self.__class__.__name__)

        self.depth_max = 80  # optional clamp (meters). Set -1 to disable in threshold_depth_map.

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: BlendedMVS scene count: {self.sequence_list_len}")
        logging.info(f"{status}: BlendedMVS dataset length: {len(self)}")

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
            seq_name (str): Name of the sequence (24-char folder name).
            ids (list): Specific IDs (frame indices) to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Determine number of frames by counting cam files
        depth_files = sorted([p for p in glob.glob(osp.join(self.BLENDED_DIR, seq_name, 'rendered_depth_maps', "*.pfm"))])
        num_images = len(depth_files)

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

        # Calculate the max / min / mean of the depth map
        depth_map_max = 0
        depth_map_min = 0
        depth_map_mean = 0

        # Read All file of images

        for local_idx in ids:
            depth_filepath = depth_files[local_idx]
            image_filepath = depth_filepath.replace("rendered_depth_maps", "blended_images").replace(".pfm", ".jpg")
            cam_filepath = depth_filepath.replace("rendered_depth_maps", "cams").replace(".pfm", "_cam.txt")
            # depth_filepath = osp.join(self.BLENDED_DIR, seq_name, 'rendered_depth_maps', f'{stem}.pfm')
            # cam_filepath = osp.join(self.BLENDED_DIR, seq_name, 'cams', f'{stem}_cam.txt')

            image = read_image_cv2(image_filepath)
            depth_map = read_depth(depth_filepath, scale_adjustment=1.0)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
            depth_map_max = max(depth_map_max, depth_map.max())
            depth_map_min = min(depth_map_min, depth_map.min())
            depth_map_mean = depth_map_mean + depth_map.mean()

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            with open(cam_filepath, 'r') as f:
                extri44 = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
                assert extri44.shape == (4, 4)
                intri33 = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
                assert intri33.shape == (3, 3)

            extri_opencv = extri44[:3, :]  # camera-from-world (OpenCV), 3x4
            intri_opencv = intri33.copy()
            # Change to here

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
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        depth_map_mean = depth_map_mean / len(ids)
        print(f"Depth map max: {depth_map_max}, min: {depth_map_min}, mean: {depth_map_mean}")

        set_name = "blendedmvs"
        batch = {
            "seq_name": set_name + "_" + seq_name,
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

# In training folder, run python data/datasets/blendedmvs.py
if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    dataset = BlendedMVSDataset(common_conf=cfg.data.train.common_config, split="train", BLENDED_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/BlendedMVS/BlendedMVS")
    print(dataset[(0, 24, 1.0)])