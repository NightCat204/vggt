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
from utils.scannet_sens_reader import load_sens_file

from hydra import initialize, compose


class ScanNetv2(BaseDataset):
    """
    ScanNetv2 Dataset Loader
    
    Reads RGB-D data directly from .sens files using scannet_sens_reader.
    Compatible with VGGT training framework.
    
    Dataset structure:
        ScanNetv2_DIR/
            scene0000_00/
                scene0000_00.sens  <- Main data file
                scene0000_00.txt   <- Scene metadata (optional)
            scene0001_00/
                ...
    """
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ScanNetv2_DIR: str = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        max_frames_per_scene: int = None,  # Load all frames if None
        use_color_intrinsics: bool = True,  # Use color camera intrinsics (vs depth)
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
        self.max_frames_per_scene = max_frames_per_scene
        self.use_color_intrinsics = use_color_intrinsics

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
            
        logging.info(f"ScanNetv2 Dataset Dir = {self.ScanNetv2_DIR}")

        # Scan for .sens files instead of processed directories
        sens_files = glob.glob(osp.join(self.ScanNetv2_DIR, "*/*.sens"))
        sens_files = sorted(sens_files)
        
        logging.info(f"Found {len(sens_files)} .sens files")

        # Cache to store loaded sensor data (avoid repeated file I/O)
        self._sensor_data_cache = {}
        
        # Filter scenes with enough frames by reading .sens headers
        filtered_list = []
        seq_lengths = []
        
        for sens_file in sens_files:
            scene_name = osp.basename(osp.dirname(sens_file))
            
            try:
                # Load sensor data to check frame count
                sensor_data = self._load_sensor_data(sens_file)
                num_frames = len(sensor_data.color_images)
                
                if num_frames >= self.min_num_images:
                    filtered_list.append(sens_file)
                    seq_lengths.append(num_frames)
                    logging.debug(f"Scene {scene_name}: {num_frames} frames")
                else:
                    logging.debug(f"Skipping {scene_name}: only {num_frames} frames (< {self.min_num_images})")
                    
            except Exception as e:
                logging.warning(f"Failed to load {sens_file}: {e}")
                continue

        self.sequence_list = filtered_list
        self.sequence_list_len = len(self.sequence_list)

        # Save statistics
        if self.sequence_list_len > 0:
            scene_names = [osp.basename(osp.dirname(f)) for f in self.sequence_list]
            self.save_seq_stats(scene_names, seq_lengths, self.__class__.__name__)

        self.depth_max = 10.0  # Max depth in meters (ScanNet indoor scenes)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ScanNetv2 scene count: {self.sequence_list_len}")
        logging.info(f"{status}: ScanNetv2 dataset length: {len(self)}")
    
    def _load_sensor_data(self, sens_file: str):
        """
        Load and cache sensor data from .sens file
        
        Args:
            sens_file: Path to .sens file
            
        Returns:
            SensorData object
        """
        if sens_file not in self._sensor_data_cache:
            logging.info(f"Loading SENS file: {sens_file}")
            sensor_data = load_sens_file(sens_file, max_frames=self.max_frames_per_scene)
            self._sensor_data_cache[sens_file] = sensor_data
        
        return self._sensor_data_cache[sens_file]

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,   
        ids: list = None,       
        aspect_ratio: float = 1.0
    ) -> dict:
        """
        Retrieve data for a specific scene from .sens file
        
        Args:
            seq_index: Index of sequence in sequence_list
            img_per_seq: Number of images to sample per sequence
            seq_name: Not used (kept for API compatibility)
            ids: Specific frame indices to load
            aspect_ratio: Target aspect ratio for image processing
            
        Returns:
            Dictionary containing:
                - seq_name: Scene name
                - ids: Frame indices
                - frame_num: Number of frames
                - images: List of RGB images (H, W, 3)
                - depths: List of depth maps (H, W)
                - extrinsics: List of camera extrinsics (3, 4) [R|t] camera-to-world
                - intrinsics: List of camera intrinsics (3, 3)
                - cam_points: List of 3D points in camera coordinates
                - world_points: List of 3D points in world coordinates
                - point_masks: List of valid point masks
                - original_sizes: List of original image sizes
        """
        # Random sampling if training
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        # Get .sens file path
        sens_file = self.sequence_list[seq_index]
        scene_name = osp.basename(osp.dirname(sens_file))
        
        # Load sensor data (cached)
        sensor_data = self._load_sensor_data(sens_file)
        num_images = len(sensor_data.color_images)

        # Sample frame indices
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        # Apply nearby sampling if enabled
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        # Get target image shape
        target_image_shape = self.get_target_shape(aspect_ratio)

        # Get camera intrinsics (use color or depth camera)
        if self.use_color_intrinsics:
            intrinsic_matrix = sensor_data.get_color_intrinsics_3x3()
        else:
            intrinsic_matrix = sensor_data.get_depth_intrinsics_3x3()

        # Initialize output lists
        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points = [], []
        point_masks, original_sizes = [], []

        # Process each frame
        for local_idx in ids:
            # Get frame data from sensor_data
            color_img, depth_img, pose_c2w = sensor_data.get_frame_data(local_idx)
            
            if color_img is None or depth_img is None:
                logging.warning(f"Skipping frame {local_idx} in {scene_name} (invalid data)")
                continue
            
            # Convert depth from uint16 (millimeters) to float32 (meters)
            depth_m = depth_img.astype(np.float32) / 1000.0
            
            # Apply depth thresholding
            depth_m = threshold_depth_map(
                depth_m, 
                max_percentile=-1, 
                min_percentile=-1, 
                max_depth=self.depth_max
            )
            
            original_size = np.array(color_img.shape[:2])
            
            # Convert camera-to-world to world-to-camera (OpenCV convention)
            # pose_c2w is the camera-to-world transformation from SENS file
            # We need world-to-camera (camera-from-world) for VGGT
            pose_w2c = np.linalg.inv(pose_c2w)
            extri_opencv = pose_w2c[:3, :]  # (3, 4) [R|t]
            intri_opencv = intrinsic_matrix.copy()

            # Process the image using base class method
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
                color_img,
                depth_m,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=f"{scene_name}/frame_{local_idx}",
            )

            # Validate output shape
            if (image.shape[:2] != target_image_shape).any():
                logging.error(
                    f"Wrong shape for {scene_name} frame {local_idx}: "
                    f"expected {target_image_shape}, got {image.shape[:2]}"
                )
                continue

            # Append to output lists
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # Construct output batch
        set_name = "scanNetv2"
        batch = {
            "seq_name": f"{set_name}_{scene_name}",
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
    # Test loading ScanNetv2 from .sens files
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    
    # ⚠️ PATH TO MODIFY: Update ScanNetv2_DIR to your dataset location
    dataset = ScanNetv2(
        common_conf=cfg.data.train.common_config, 
        split="train", 
        ScanNetv2_DIR="/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2",  # <- CHANGE THIS
        max_frames_per_scene=50,  # Limit frames for faster testing
    )
    
    print(f"Dataset initialized with {len(dataset.sequence_list)} scenes")
    
    # Test getting data
    if len(dataset.sequence_list) > 0:
        batch = dataset.get_data(seq_index=0, img_per_seq=5, aspect_ratio=1.0)
        print(f"Loaded batch: {batch['seq_name']}")
        print(f"  Frames: {batch['frame_num']}")
        print(f"  Image shape: {batch['images'][0].shape}")
        print(f"  Depth shape: {batch['depths'][0].shape}")
        print(f"  Intrinsics shape: {batch['intrinsics'][0].shape}")
        print(f"  Extrinsics shape: {batch['extrinsics'][0].shape}")





