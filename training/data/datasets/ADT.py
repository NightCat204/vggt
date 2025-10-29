import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np
import json


from data.dataset_util import *
from data.base_dataset import BaseDataset

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinDataPathsProvider,
   utils as adt_utils,
)
import logging
logging.getLogger("projectaria_tools").setLevel(logging.WARNING)

from hydra import initialize, compose


class ADTDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ADT_DIR: str = "/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ADT",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Dataset format:
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.ADT_DIR = ADT_DIR
        self.metadata_file = os.path.join(ADT_DIR, "metadata.json")
        self.min_num_images = min_num_images
        self._provider_cache = {} # cache provider for the same sequence

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"ADT_DIR is {self.ADT_DIR}")

        if not osp.exists(self.metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}\n"
                f"Please run preprocess_adt_metadata.py first:\n"
                f"  python preprocess_adt_metadata.py --adt-dir {self.ADT_DIR}"
            )

        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)

        # Load sequence list
        sequence_list = []
        sequence_lengths = []

        for seq in metadata['sequences']:
            if seq['num_images'] <= self.min_num_images:
                continue
            
            # get info from metadata
            seq_info = {
                "seq_Dir": seq['seq_Dir'],
                "scene_name": seq['scene_name'],
                "stream_id_str": seq['stream_id'],
                "num_images": seq['num_images'],
                "img_timestamps_ns_all": np.array(seq['timestamps_ns'], dtype=np.int64),
                "intrinsics": seq['intrinsics'],
                "T_Device_Cam": np.array(seq['T_Device_Cam'], dtype=np.float64),
                "paths": seq['paths'], # original path include :"video_vrs","depth_vrs","trajectory_csv"

                # do not add to seq_info to avoid loading too much data
                "gt_provider": None,
                "data_paths": None,

            }
            sequence_list.append(seq_info)
            sequence_lengths.append(seq_info['num_images'])

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        # Analyze the sequence number and the distribution of the sequence length
        self.save_seq_stats(self.sequence_list, sequence_lengths, self.__class__.__name__)

        self.depth_max = 80  # optional clamp (meters). Set -1 to disable in threshold_depth_map.

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ADT scene count: {self.sequence_list_len}")
        logging.info(f"{status}: ADT dataset length: {len(self)}")

    def _get_or_create_provider(self, seq_Dir: str):
        """
        获取或创建provider，使用缓存避免重复创建
        同一个场景的不同stream共享同一个provider
        """
        if seq_Dir not in self._provider_cache:
            paths_provider = AriaDigitalTwinDataPathsProvider(seq_Dir)
            data_paths = paths_provider.get_datapaths()
            gt_provider = AriaDigitalTwinDataProvider(data_paths)
            self._provider_cache[seq_Dir] = gt_provider
        
        return self._provider_cache[seq_Dir]
    
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,   
        ids: list = None,       
        aspect_ratio: float = 1.0,
    ) -> dict:
        
        info = self.sequence_list[seq_index]
        seq_Dir = info['seq_Dir']
        img_timestamps_ns_all = info['img_timestamps_ns_all']
        scene_name = info['scene_name']
        stream_id = StreamId(info['stream_id_str'])
        img_timestamps_ns_all = info['img_timestamps_ns_all']
        T_Device_Cam = info['T_Device_Cam'] # 4x4 matrix
        gt_provider = self._get_or_create_provider(seq_Dir)

        
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = scene_name + f"_{stream_id}"

        if ids is None:
            num_images = info['num_images']
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points = [], []
        point_masks, original_sizes = [], []

        intrinsics_params = info['intrinsics']
        fx = intrinsics_params['fx']
        fy = intrinsics_params['fy']
        cx = intrinsics_params['cx']
        cy = intrinsics_params['cy']

        K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        
        for local_idx in ids:
            timestamp = img_timestamps_ns_all[local_idx]
            
            image = gt_provider.get_aria_image_by_timestamp_ns(timestamp, stream_id).data().to_numpy_array() # gray
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # image shape  = 2 
            original_size = np.array(image.shape[:2]) 

            # 单位为毫米
            # example: Depth max: 8434.0, min: 0, mean: 3013.924082438151
            depth_map = gt_provider.get_depth_image_by_timestamp_ns(timestamp, stream_id).data().to_numpy_array()
            # depth_map = depth_map / 1000.0 # convert to meters
            depth_map = threshold_depth_map(depth_map)  # not in meters 

            # extrinsics
            aria_pose = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp).data()
            T_Scene_Device = aria_pose.transform_scene_device.to_matrix() # sophus.SE3 to matrix
            T_Scene_Cam = T_Scene_Device @ T_Device_Cam # Cam to Scene
            T_cam_scene = np.linalg.inv(T_Scene_Cam) # Scene to Cam
            extri_opencv = T_cam_scene[:3, :]

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
                filepath=None, # no image file path
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

        set_name = "ADT"
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
    dataset = ADTDataset(common_conf=cfg.data.train.common_config, split="train", ADT_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ADT")
    print(dataset[(0, 24, 1.0)])