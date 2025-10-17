# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import os.path as osp
import logging
import random
import glob
import json

import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.dataset_util import *
from data.base_dataset import BaseDataset
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf  # noqa: F401


class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = "/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/DL3DV-ALL-960P/DL3DV",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the DL3DVDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): 'train' or 'test'.
            DL3DV_DIR (str): Root path of DL3DV dataset.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_ratio (int): expand range used by get_nearby_ids.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.DL3DV_DIR = DL3DV_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"DL3DV_DIR is {self.DL3DV_DIR}")

        txt_path = osp.join(self.DL3DV_DIR, "sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, "r") as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            json_paths = glob.glob(osp.join(self.DL3DV_DIR, "*", "transforms.json"))
            sequence_list = [p.split(self.DL3DV_DIR)[-1].lstrip("/") for p in json_paths]
            sequence_list = sorted(sequence_list)
            with open(txt_path, "w") as f:
                f.write("\n".join(sequence_list))

        filtered = []
        for rel_json in sequence_list:
            full_json = osp.join(self.DL3DV_DIR, rel_json)
            try:
                with open(full_json, "r") as f:
                    meta = json.load(f)
                frames = meta.get("frames", [])
                if len(frames) >= self.min_num_images:
                    filtered.append(rel_json)
            except Exception as e:
                logging.warning(f"Skip invalid transforms.json: {full_json} ({e})")
        self.sequence_list = filtered
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: DL3DV sequences: {self.sequence_list_len}")
        logging.info(f"{status}: DL3DV dataset length: {len(self)}")

        self._seq_meta_cache = {}


    def _load_seq_meta(self, seq_rel_json_path: str):
        """Load and cache transforms.json along with resolved frame paths and intrinsics."""
        if seq_rel_json_path in self._seq_meta_cache:
            return self._seq_meta_cache[seq_rel_json_path]

        full_json = osp.join(self.DL3DV_DIR, seq_rel_json_path)
        with open(full_json, "r") as f:
            meta = json.load(f)

        fx = float(meta.get("fl_x"))
        fy = float(meta.get("fl_y"))
        cx = float(meta.get("cx"))
        cy = float(meta.get("cy"))
        intri_opencv = np.eye(3, dtype=np.float32)
        intri_opencv[0, 0] = fx
        intri_opencv[1, 1] = fy
        intri_opencv[0, 2] = cx
        intri_opencv[1, 2] = cy

        seq_dir_rel = osp.dirname(seq_rel_json_path)  # e.g. <hash>
        seq_dir_abs = osp.join(self.DL3DV_DIR, seq_dir_rel)

        frames = []
        for fr in meta.get("frames", []):
            rel_img = fr.get("file_path")  # e.g. images/frame_00001.png
            img_abs = osp.join(seq_dir_abs, rel_img)
            if not osp.exists(img_abs):
                continue
            T = np.array(fr["transform_matrix"], dtype=np.float32)  # 4x4
            extri = T[:3, :]  # 3x4
            frames.append((img_abs, extri))
        
        meta_w = int(meta.get("w"))
        meta_h = int(meta.get("h"))

        seq_meta = {
            "seq_dir_rel": seq_dir_rel,
            "seq_dir_abs": seq_dir_abs,
            "intrinsics": intri_opencv,   
            "meta_wh": (meta_w, meta_h),  
            "frames": frames,
        }

        self._seq_meta_cache[seq_rel_json_path] = seq_meta
        return seq_meta

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific DL3DV sequence.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_rel_json = self.sequence_list[seq_index]
        else:
            raise ValueError(f"seq_name must be None when seq_index is provided")

        seq_meta = self._load_seq_meta(seq_rel_json)
        frames = seq_meta["frames"]
        num_images = len(frames)
        if num_images < 1:
            raise RuntimeError(f"No valid frames in sequence: {seq_rel_json}")

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        # print(f"target_image_shape: {target_image_shape}")

        images, depths = [], []
        cam_points, world_points, point_masks = [], [], []
        extrinsics, intrinsics, original_sizes = [], [], []

        intri_opencv_template = seq_meta["intrinsics"]
        meta_w, meta_h = seq_meta["meta_wh"]

        for image_idx in ids:
            img_path, extri_opencv = frames[int(image_idx)]

            image = read_image_cv2(img_path)
            H, W = image.shape[:2]

            # 从 4K(meta_w, meta_h) 缩放到当前图像分辨率 (W, H)
            sx = W / float(meta_w)
            sy = H / float(meta_h)

            intri_opencv = intri_opencv_template.copy()
            intri_opencv[0, 0] *= sx  # fx
            intri_opencv[1, 1] *= sy  # fy
            intri_opencv[0, 2] *= sx  # cx
            intri_opencv[1, 2] *= sy  # cy

            depth_map = np.ones((H, W), dtype=np.float32)
            original_size = np.array([H, W], dtype=np.int32)

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
                filepath=img_path,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(
                    f"Wrong shape for {seq_rel_json}: expected {target_image_shape}, got {image.shape[:2]}"
                )
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "dl3dv"
        batch = {
            "seq_name": set_name + "_" + seq_meta["seq_dir_rel"],
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
    dataset = DL3DVDataset(common_conf=cfg.data.train.common_config, split="train", DL3DV_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/DL3DV-ALL-960P/DL3DV")
    print(dataset[(0, 24, 0.8)])

