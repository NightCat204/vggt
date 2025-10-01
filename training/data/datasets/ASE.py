import os
import sys
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset

from projectaria_tools.projects import ase
from data.utils import read_trajectory_file

from hydra import initialize, compose


class ASEDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ASE_DIR: str = "/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ASE",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        Dataset format:
        image: <scenID>/rgb/vignette<index>.jpg/
        depth: <scenID>/depth/depth<index>.png mm为单位
        trajectory: <scenID>/trajectory.csv
        calibration: same for the whole ASE dataset, load from get_ase_rgb_calibration()
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.ASE_DIR = ASE_DIR
        self.min_num_images = min_num_images

        # unified calibration for all ASE sequences, save in self
        self.calib_device = ase.get_ase_rgb_calibration()
        self.projection_params = self.calib_device.get_projection_params()

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"ASE_DIR is {self.ASE_DIR}")

        # Load sequence list(from 0 to 99)
        sequence_list = [p for p in glob.glob(osp.join(self.ASE_DIR, "*")) if osp.isdir(p)]
        sequence_list = sorted(sequence_list)
        
        # Filter sequences with enough images available
        filtered_list = []
        seq_lengths = []
        for seq in sequence_list:
            img_dir = osp.join(self.ASE_DIR, seq, 'rgb')
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
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Determine number of frames by counting image files
        img_dir = osp.join(seq_name, 'rgb')
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

        # read extrinsics from trajectory csv
        traj_path = osp.join(seq_name, 'trajectory.csv')
        trajectory = read_trajectory_file(traj_path)

        # intrinsics from projection params
        # data from the source sode: 
        # Eigen::VectorXd projectionParams(projectaria::tools::calibration::Fisheye624::kNumParams);
        # projectionParams << 297.6375381033778, 357.6599197217746, 349.1922497127481, 0.3650890375644368,
        # -0.1738082418112771, -0.7534945484033189, 2.434788882752295, -2.57786220300886,
        # 0.8788483538598834, 0.0008005198595407136, -0.000294237814554143, 0., 0., 0., 0.;

        # structure of the params comes from meta document
        fx = self.projection_params[0]
        fy = self.projection_params[0]
        cx = self.projection_params[1]
        cy = self.projection_params[2]

        K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

        for local_idx in ids:
            # 7 digit path format: vignette0000000.jpg (starting from 0)
            frame_id = str(local_idx).zfill(7)
            image_filepath = osp.join(seq_name, 'rgb', f'vignette{frame_id}.jpg')
            depth_filepath = osp.join(seq_name, 'depth', f'depth{frame_id}.png')

            image = read_image_cv2(image_filepath)
            # depth has a problem
            # As stated in the doc, the pixel contents are integers expressing the depth along the pixel’s ray direction, in units of mm.
            # but checked result very small
            # example: max: 0.0007991790771484375
            depth_map = read_depth(depth_filepath) # 16-bit png depth
            depth_map = threshold_depth_map(depth_map)  # not in meters 

            original_size = np.array(image.shape[:2])

            T_world_from_device = trajectory["Ts_world_from_device"][local_idx]
            T_device_from_world = np.linalg.inv(T_world_from_device)
            T_device_from_camera = self.calib_device.get_transform_device_camera().to_matrix()
            T_camera_from_device = np.linalg.inv(T_device_from_camera)
            T_camera_from_world = T_camera_from_device @ T_device_from_world
            extri_opencv = T_camera_from_world[:3, :]
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

        set_name = "ASE"
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
    dataset = ASEDataset(common_conf=cfg.data.train.common_config, split="train", ASE_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/ASE")
    print(dataset[(0, 24, 1.0)])
            