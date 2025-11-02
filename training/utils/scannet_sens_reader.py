"""
ScanNet .sens file reader for VGGT training framework
Adapted from: https://github.com/ScanNet/ScanNet/tree/master/SensReader/python

Reads compressed RGB-D sensor stream data from ScanNet .sens files.

NOTE: This file contains NO hardcoded paths - it's fully portable.
Usage: Just import and call load_sens_file() with your .sens file path.
"""

import os
import struct
import numpy as np
import zlib
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SensorData:
    """Holds RGB-D sensor stream data from a .sens file"""
    
    def __init__(self):
        self.version = 0
        self.depth_width = 0
        self.depth_height = 0
        self.color_width = 0
        self.color_height = 0
        self.depth_shift = 1000.0  # depth unit conversion factor
        self.num_frames = 0
        
        # Camera intrinsics (4x4 matrices)
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.depth_to_color_extrinsics = None
        
        # Frame data
        self.depth_images = []
        self.color_images = []
        self.camera_poses = []  # 4x4 camera-to-world matrices
        self.timestamps = []
        
    def read_header(self, f):
        """
        Read .sens file header
        
        Format (version 4):
        - version: uint (4 bytes)
        - strlen: uint64 (8 bytes)  
        - sensor_name: char[strlen]
        - intrinsic_color: float[16] (64 bytes, 4x4 matrix)
        - extrinsic_color: float[16] (64 bytes, 4x4 matrix)
        - intrinsic_depth: float[16] (64 bytes, 4x4 matrix)
        - extrinsic_depth: float[16] (64 bytes, 4x4 matrix)
        - color_compression_type: int (4 bytes)
        - depth_compression_type: int (4 bytes)
        - color_width: uint (4 bytes)
        - color_height: uint (4 bytes)
        - depth_width: uint (4 bytes)
        - depth_height: uint (4 bytes)
        - depth_shift: float (4 bytes)
        - num_frames: uint64 (8 bytes)
        """
        # Version
        self.version = struct.unpack('I', f.read(4))[0]
        logger.debug(f"SENS file version: {self.version}")
        
        # Scene name
        strlen = struct.unpack('Q', f.read(8))[0]
        scene_name = f.read(strlen).decode('utf-8')
        logger.debug(f"Scene name: {scene_name}")
        
        # Read intrinsics and extrinsics
        # Note: color comes BEFORE depth in the file format
        self.color_intrinsics = np.array(struct.unpack('f' * 16, f.read(64))).reshape(4, 4)
        self.color_extrinsics = np.array(struct.unpack('f' * 16, f.read(64))).reshape(4, 4)
        self.depth_intrinsics = np.array(struct.unpack('f' * 16, f.read(64))).reshape(4, 4)
        self.depth_to_color_extrinsics = np.array(struct.unpack('f' * 16, f.read(64))).reshape(4, 4)
        
        # Compression types
        color_compression = struct.unpack('i', f.read(4))[0]
        depth_compression = struct.unpack('i', f.read(4))[0]
        logger.debug(f"Compression - Color: {color_compression}, Depth: {depth_compression}")
        
        # Resolution: color first, then depth
        self.color_width = struct.unpack('I', f.read(4))[0]
        self.color_height = struct.unpack('I', f.read(4))[0]
        self.depth_width = struct.unpack('I', f.read(4))[0]
        self.depth_height = struct.unpack('I', f.read(4))[0]
        
        logger.debug(f"Color resolution: {self.color_width}x{self.color_height}")
        logger.debug(f"Depth resolution: {self.depth_width}x{self.depth_height}")
        
        # Depth shift
        self.depth_shift = struct.unpack('f', f.read(4))[0]
        logger.debug(f"Depth shift: {self.depth_shift}")
        
        # Number of frames
        self.num_frames = struct.unpack('Q', f.read(8))[0]
        logger.debug(f"Number of frames: {self.num_frames}")
        
    def read_frames(self, f, max_frames: Optional[int] = None):
        """Read all frame data (RGB, depth, poses)"""
        num_frames_to_read = min(self.num_frames, max_frames) if max_frames else self.num_frames
        
        logger.debug(f"Reading {num_frames_to_read} frames...")
        
        for i in range(num_frames_to_read):
            # Camera to world pose (4x4)
            pose = np.array(struct.unpack('f' * 16, f.read(64))).reshape(4, 4)
            self.camera_poses.append(pose)
            
            # Timestamps: color first, then depth
            timestamp_color = struct.unpack('Q', f.read(8))[0]
            timestamp_depth = struct.unpack('Q', f.read(8))[0]
            self.timestamps.append((timestamp_color, timestamp_depth))
            
            # Compressed image sizes: color first, then depth
            color_size = struct.unpack('Q', f.read(8))[0]
            depth_size = struct.unpack('Q', f.read(8))[0]
            
            # Compressed image data: color first, then depth
            color_compressed = f.read(color_size)
            depth_compressed = f.read(depth_size)
            
            # Decompress and decode
            try:
                # Color: decode JPEG directly (no zlib)
                color_image = cv2.imdecode(
                    np.frombuffer(color_compressed, dtype=np.uint8), 
                    cv2.IMREAD_COLOR
                )
                # Convert BGR to RGB
                if color_image is not None:
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    self.color_images.append(color_image)
                else:
                    logger.warning(f"Failed to decode color image for frame {i}")
                    self.color_images.append(None)
                
                # Depth: decompress with zlib then convert to uint16
                depth_data = zlib.decompress(depth_compressed)
                depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                    self.depth_height, self.depth_width
                )
                self.depth_images.append(depth_image)
                
            except Exception as e:
                logger.warning(f"Failed to decompress frame {i}: {e}")
                self.depth_images.append(None)
                self.color_images.append(None)
    
    def get_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific frame
        
        Returns:
            color_image: (H, W, 3) RGB uint8
            depth_image: (H, W) uint16 (in millimeters)
            pose: (4, 4) camera-to-world transformation matrix
        """
        if frame_idx >= len(self.color_images):
            raise IndexError(f"Frame {frame_idx} out of range (total: {len(self.color_images)})")
        
        return self.color_images[frame_idx], self.depth_images[frame_idx], self.camera_poses[frame_idx]
    
    def get_depth_intrinsics_3x3(self) -> np.ndarray:
        """Get 3x3 depth camera intrinsics matrix"""
        return self.depth_intrinsics[:3, :3]
    
    def get_color_intrinsics_3x3(self) -> np.ndarray:
        """Get 3x3 color camera intrinsics matrix"""
        return self.color_intrinsics[:3, :3]


def load_sens_file(sens_path: str, max_frames: Optional[int] = None) -> SensorData:
    """
    Load a ScanNet .sens file
    
    Args:
        sens_path: Path to .sens file
        max_frames: Maximum number of frames to load (None = load all)
    
    Returns:
        SensorData object with loaded data
    """
    if not os.path.exists(sens_path):
        raise FileNotFoundError(f"SENS file not found: {sens_path}")
    
    sensor_data = SensorData()
    
    with open(sens_path, 'rb') as f:
        sensor_data.read_header(f)
        sensor_data.read_frames(f, max_frames=max_frames)
    
    logger.debug(f"Successfully loaded {len(sensor_data.color_images)} frames from {sens_path}")
    return sensor_data
