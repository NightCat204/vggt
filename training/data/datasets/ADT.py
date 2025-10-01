import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset

print("Setting GLOG_minloglevel=2 to suppress INFO and WARNING logs...")
os.environ['GLOG_minloglevel'] = '2'
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinDataPathsProvider,
   utils as adt_utils,
)

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
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"ADT_DIR is {self.ADT_DIR}")

        # Load sequence list
        sequence_list = []
        sequence_lengths = []
        sequenceDir_list = [
            p for p in glob.glob(osp.join(self.ADT_DIR, "*")) 
            if osp.isdir(p) and osp.basename(p) != "projectaria_tools"
            ]
        sequenceDir_list = sorted(sequenceDir_list)

        # sequenceDir_list = sequenceDir_list[:1] # for test

        for seq_Dir in sequenceDir_list:
            paths_provider = AriaDigitalTwinDataPathsProvider(seq_Dir)
            data_paths = paths_provider.get_datapaths()

            gt_provider = AriaDigitalTwinDataProvider(data_paths)
            # 1. Raw Aria recording (stream ids: 211-1, 214-1, 247-1, 1201-1/2, 1202-1/2, 1203-1)
            # 2. Synthetic twin recording (stream ids: 214-1, 1201-1/2, 1202-2)
            stream_ids = ["214-1", "1201-1", "1201-2"]

            # load info by ft_provider
            for id in stream_ids:
                stream_id = StreamId(id)
                img_timestamps_ns_all = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
                length = len(img_timestamps_ns_all)
                if length < self.min_num_images:
                    continue
                camera_calibration = gt_provider.get_aria_camera_calibration(stream_id) # class method
                seq_info = {
                    "gt_provider": gt_provider,
                    "data_paths": data_paths,
                    "img_timestamps_ns_all": img_timestamps_ns_all,
                    "camera_calibration": camera_calibration,
                    "seq_Dir": seq_Dir, 
                    "scene_name": osp.basename(seq_Dir), 
                    "stream_id": stream_id,
                    "num_images": length,
                }
                sequence_list.append(seq_info)
                sequence_lengths.append(length)

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        # Analyze the sequence number and the distribution of the sequence length
        self.save_seq_stats(self.sequence_list, sequence_lengths, self.__class__.__name__)

        self.depth_max = 80  # optional clamp (meters). Set -1 to disable in threshold_depth_map.

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ADT scene count: {self.sequence_list_len}")
        logging.info(f"{status}: ADT dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,   
        ids: list = None,       
        aspect_ratio: float = 1.0,
    ) -> dict:
        
        info = self.sequence_list[seq_index]
        gt_provider = info['gt_provider']
        data_paths = info['data_paths']
        img_timestamps_ns_all = info['img_timestamps_ns_all']
        calib = info['camera_calibration']
        scene_name = info['scene_name']
        stream_id = info['stream_id']
        seq_Dir = info['seq_Dir']
        
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

        traj_path = data_paths.aria_trajectory_filepath

        (cx,cy) = calib.get_principal_point()
        (fx,fy) = calib.get_focal_lengths()

        K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        
        for local_idx in ids:
            timestamp = img_timestamps_ns_all[local_idx]
            
            image = gt_provider.get_aria_image_by_timestamp_ns(timestamp, stream_id).data().to_numpy_array() # rgb
            original_size = np.array(image.shape[:2]) # shape (1408, 1408, 3)

            depth_map = gt_provider.get_depth_image_by_timestamp_ns(timestamp, stream_id).data().to_numpy_array()
            
            # extrinsics
            T_Device_Cam = calib.get_transform_device_camera()
            aria_pose = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp).data()
            T_Scene_Device = aria_pose.transform_scene_device
            T_Scene_Cam = T_Scene_Device @ T_Device_Cam # Cam to Scene, sophus.SE3 
            T_cam_scene = np.linalg.inv(T_Scene_Cam.to_matrix()) # Scene to Cam
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