import os
import os.path as osp
import glob
import logging
import random
import json
from pathlib import Path

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset

from hydra import initialize, compose


class ScanNetv2(BaseDataset):
    '''
   
    '''
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ScanNetv2_DIR: str = "/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/vlm_visual_odom/ScanNet/decoded/test/",
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
        self.ScanNetv2_DIR = ScanNetv2_DIR.rstrip("/")
        self.min_num_images = min_num_images


        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
            
        logging.info(f"ScanNetv2 Dataset Dir = {self.ScanNetv2_DIR}")

        sequence_list = [p for p in glob.glob(osp.join(self.ScanNetv2_DIR, "*")) if osp.isdir(p)]
        sequence_list = sorted(sequence_list)

        # Filter sequences with enough images available
        filtered_list = []
        seq_lengths = []
        for seq in sequence_list:
            img_dir = osp.join(self.ScanNetv2_DIR, seq, 'color')
            img_files = sorted([p for p in glob.glob(osp.join(img_dir, "*")) if osp.basename(p).endswith('.jpg')])
            if len(img_files) >= self.min_num_images:
                filtered_list.append(seq)
                seq_lengths.append(len(img_files))

        self.sequence_list = filtered_list
        self.sequence_list_len = len(self.sequence_list)

        # Analyze the sequence number and the distribution of the sequence length
        self.save_seq_stats(self.sequence_list, seq_lengths, self.__class__.__name__)

        self.depth_max = 80  # optional clamp (meters). Set -1 to disable in threshold_depth_map.

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ASE scene count: {self.sequence_list_len}")
        logging.info(f"{status}: ASE dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,   
        ids: list = None,       
        aspect_ratio: float = 1.0
    ) -> dict:

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Determine number of frames by counting image files
        img_dir = osp.join(seq_name, 'color')
        img_files = sorted(glob.glob(osp.join(img_dir, '*.jpg')))
        num_images = len(img_files)

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points = [], []
        point_masks, original_sizes = [], []

        intrinsics_txt_dir = osp.join(seq_name, 'intrinsic', 'intrinsic_color.txt')
        intrinsic_matrix = np.loadtxt(intrinsics_txt_dir)

        for local_idx in ids:
            # temporily set as five step
            frame_id = str(local_idx * 5)
            image_filepath = osp.join(seq_name, 'color', f'{frame_id}.jpg')
            image = read_image_cv2(image_filepath)
            
            depth_filepath = osp.join(seq_name, 'depth',f'{frame_id}.jpg')
            depth_map = read_depth(depth_filepath)
         
            # for debuging, use fake depth
            # depth_map = np.full(image.shape[:2], 5.0, dtype=np.float32) # update this after download
            depth_map = threshold_depth_map(depth_map)

            original_size = np.array(image.shape[:2])

            pose_file_path = osp.join(seq_name, 'pose', f'{frame_id}.txt')
            extrinsic_matrix = np.loadtxt(pose_file_path)
            
            extri_opencv = extrinsic_matrix[:3, :]
            intri_opencv = intrinsic_matrix[:3, :3].copy()

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

        set_name = "scanNetv2"
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

if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    dataset = ScanNetv2(common_conf=cfg.data.train.common_config, split="train", ScanNetv2_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/vlm_visual_odom/ScanNet/decoded/test/")
    print(dataset[(0, 24, 1.0)])





