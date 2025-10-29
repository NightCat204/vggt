import os
import os.path as osp
import logging
import random
import pickle
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

from data.dataset_util import read_image_cv2, threshold_depth_map, read_depth
from data.base_dataset import BaseDataset
from PIL import Image

class MapillaryDataset(BaseDataset):
    DIRS = ["FRONT", "LEFT", "RIGHT", "BACK"]
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MAPILLARY_DIR: str = "/path/to/Mapillary",
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
        self.MAPILLARY_DIR = MAPILLARY_DIR
        self.min_num_images = int(min_num_images)
        self.depth_max = 80.0

        if split == "train":
            self.len_train = int(len_train)
        elif split == "test":
            self.len_train = int(len_test)
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"Loading Mapillary sequence index: {self.MAPILLARY_DIR}")

        seq_path = osp.join(self.MAPILLARY_DIR, "sequence_list.pt")
        self.seq_map = self._load_seq_map(seq_path)  
        assert len(self.seq_map) > 0, "Empty or invalid sequence_list.pt"

        allow = set(self.DIRS)

        self.sequence_list: List[Tuple[str, str]] = []
        self._seq_len_cache: List[int] = []

        for scene_token, dir_dict in self.seq_map.items():
            if not isinstance(dir_dict, dict):
                continue
            for d, items in dir_dict.items():
                du = str(d).upper()
                if du not in allow:
                    continue
                if not isinstance(items, list) or len(items) < self.min_num_images:
                    continue
                self.sequence_list.append((scene_token, du))
                self._seq_len_cache.append(len(items))

        self.sequence_list_len = len(self.sequence_list)
        self.save_seq_stats(
            [f"{st}/{d}" for st, d in self.sequence_list], self._seq_len_cache, self.__class__.__name__
        )

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Mapillary scene_direction count: {self.sequence_list_len}")
        logging.info(f"{status}: Mapillary dataset length: {len(self)}")


    def _load_seq_map(self, pt_path: str) -> Dict[str, Dict[str, List[dict]]]:
        if not osp.isfile(pt_path):
            raise FileNotFoundError(f"sequence_list.pt not found: {pt_path}")

        if torch is not None:
            try:
                data = torch.load(pt_path, map_location="cpu")
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logging.warning(f"torch.load failed ({e}), fallback to pickle.")

    def _get_seq_items(self, scene_token: str, direction: str) -> List[dict]:
        dir_dict = self.seq_map.get(scene_token, {})
        items = dir_dict.get(direction, [])
        if not isinstance(items, list):
            return []
        return items


    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,   # "scene_token/direction"
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            scene_token, direction = self.sequence_list[seq_index]
        else:
            if "/" in seq_name:
                scene_token, direction = seq_name.split("/", 1)
                direction = direction.upper()
            else:
                raise ValueError(f"Invalid sequence name: {seq_name}")

        items = self._get_seq_items(scene_token, direction)
        num_images = len(items)
        if num_images < 1:
            raise ValueError(f"No frames for {scene_token}/{direction}")

        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            # print(f"Getting nearby ids for {scene_token}/{direction}")
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        cam_points, world_points, point_masks = [], [], []
        extrinsics, intrinsics, original_sizes = [], [], []

        for idx in ids:
            rec = items[int(idx)]
            rgb_path   = osp.join(self.MAPILLARY_DIR, rec["rgb"])
            depth_path = osp.join(self.MAPILLARY_DIR, rec["depth"])
            intri_opencv = rec["intrinsics"].copy()
            extri_opencv = rec["extrinsics_wc"].copy()
            image = read_image_cv2(rgb_path)
            depth_map = Image.open(depth_path)
            depth_map = np.array(depth_map, dtype=np.float32) / 256.0

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2], dtype=np.int32)
            depth_map = threshold_depth_map(
                depth_map,
                max_percentile=99, min_percentile=1,
                max_depth=self.depth_max
            )

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
                logging.error(f"Wrong shape for {scene_token}/{direction}: "
                              f"expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "mapillary"
        batch = {
            "seq_name": f"{set_name}_{scene_token}_{direction}",
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
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")

    ds = MapillaryDataset(
        common_conf=cfg.data.train.common_config,
        split="train",
        SEQLIST_PT_PATH="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/Mapillary/",
    )

    print(ds[(0, 24, 1.0)])
