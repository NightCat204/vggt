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


class ScanNetv2(BaseDataset):
    """
    ScanNetv2 Dataset Loader
    
    读取预处理后的ScanNet数据（RGB/depth/pose分离存储）
    与vkitti数据集处理流程保持一致
    
    预处理数据结构（通过scannet_preprocess.py生成）：
        ScanNetv2_DIR/
            scene0000_00/
                color/
                    000000.jpg
                    000001.jpg
                    ...
                depth/
                    000000.png  (uint16, millimeters)
                    000001.png
                    ...
                pose/
                    000000.txt  (4x4 camera-to-world matrix)
                    000001.txt
                    ...
                intrinsic.txt   (3x3 camera intrinsics)
                metadata.txt    (scene metadata)
            scene0001_00/
                ...
    """
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ScanNetv2_DIR: str = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2_Processed",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        """
        初始化ScanNetv2数据集
        
        Args:
            common_conf: 通用配置对象
            split: 数据集划分 ('train' 或 'test')
            ScanNetv2_DIR: 预处理后的数据目录
            min_num_images: 每个场景最小帧数
            len_train: 训练集长度
            len_test: 测试集长度
            expand_ratio: nearby采样的扩展比例
        """
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

        # 扫描所有场景目录（含color子目录的即为有效场景）
        scene_dirs = sorted(glob.glob(osp.join(self.ScanNetv2_DIR, "*")))
        scene_dirs = [d for d in scene_dirs if osp.isdir(d) and osp.exists(osp.join(d, "color"))]
        
        logging.info(f"Found {len(scene_dirs)} potential scenes")

        # 过滤：检查帧数是否满足最小要求
        filtered_scenes = []
        seq_lengths = []
        
        for scene_dir in scene_dirs:
            scene_name = osp.basename(scene_dir)
            color_files = sorted(glob.glob(osp.join(scene_dir, "color", "*.jpg")))
            num_frames = len(color_files)
            
            if num_frames >= self.min_num_images:
                filtered_scenes.append(scene_name)
                seq_lengths.append(num_frames)
                logging.debug(f"Scene {scene_name}: {num_frames} frames")
            else:
                logging.debug(f"Skipping {scene_name}: only {num_frames} frames (< {self.min_num_images})")

        self.sequence_list = filtered_scenes
        self.sequence_list_len = len(self.sequence_list)

        # 保存统计信息
        if self.sequence_list_len > 0:
            self.save_seq_stats(self.sequence_list, seq_lengths, self.__class__.__name__)

        # ScanNet室内场景深度范围通常在10米内
        self.depth_max = 10.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ScanNetv2 scene count: {self.sequence_list_len}")
        logging.info(f"{status}: ScanNetv2 dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = 24,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        获取指定序列的数据
        
        Args:
            seq_index: 序列索引
            img_per_seq: 每个序列采样的图像数量
            seq_name: 序列名称（可选，优先使用seq_index）
            ids: 指定的帧索引列表（可选）
            aspect_ratio: 目标图像宽高比
            
        Returns:
            包含以下字段的字典：
                - seq_name: 场景名称
                - ids: 帧索引列表
                - frame_num: 帧数
                - images: RGB图像列表 (H, W, 3)
                - depths: 深度图列表 (H, W)
                - extrinsics: 外参矩阵列表 (3, 4) [R|t] world-to-camera
                - intrinsics: 内参矩阵列表 (3, 3)
                - cam_points: 相机坐标系3D点列表
                - world_points: 世界坐标系3D点列表
                - point_masks: 有效点掩码列表
                - original_sizes: 原始图像尺寸列表
        """
        # 训练时随机采样
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        # 获取场景名称
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        scene_dir = osp.join(self.ScanNetv2_DIR, seq_name)
        
        # 加载相机内参
        try:
            intrinsic_matrix = np.loadtxt(osp.join(scene_dir, "intrinsic.txt"))
        except Exception as e:
            logging.error(f"Error loading intrinsic for {seq_name}: {e}")
            raise

        # 统计可用帧数
        color_files = sorted(glob.glob(osp.join(scene_dir, "color", "*.jpg")))
        num_images = len(color_files)

        # 采样帧索引
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        # 应用nearby采样策略
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        # 获取目标图像尺寸
        target_image_shape = self.get_target_shape(aspect_ratio)

        # 初始化输出列表
        images, depths = [], []
        extrinsics, intrinsics = [], []
        cam_points, world_points = [], []
        point_masks, original_sizes = [], []

        # 逐帧加载数据
        for image_idx in ids:
            # 构建文件路径
            color_path = osp.join(scene_dir, "color", f"{image_idx:06d}.jpg")
            depth_path = osp.join(scene_dir, "depth", f"{image_idx:06d}.png")
            pose_path = osp.join(scene_dir, "pose", f"{image_idx:06d}.txt")

            # 检查文件存在性
            if not all(osp.exists(p) for p in [color_path, depth_path, pose_path]):
                logging.warning(f"Missing files for {seq_name} frame {image_idx}, skipping")
                continue

            # 读取RGB图像
            image = read_image_cv2(color_path)
            
            # 读取深度图 (uint16, millimeters)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            
            # 深度单位转换: millimeters -> meters
            depth_map = depth_map.astype(np.float32) / 1000.0
            
            # ScanNet特性：RGB和深度分辨率不同，需要resize深度图到RGB尺寸
            if image.shape[:2] != depth_map.shape:
                # 使用最近邻插值保持深度值准确性
                depth_map = cv2.resize(
                    depth_map, 
                    (image.shape[1], image.shape[0]),  # (width, height)
                    interpolation=cv2.INTER_NEAREST
                )
            
            # 深度阈值处理
            depth_map = threshold_depth_map(
                depth_map, 
                max_percentile=-1, 
                min_percentile=-1, 
                max_depth=self.depth_max
            )

            original_size = np.array(image.shape[:2])

            # 加载相机位姿 (4x4 camera-to-world)
            pose_c2w = np.loadtxt(pose_path)
            
            # 转换为world-to-camera (OpenCV约定)
            pose_w2c = np.linalg.inv(pose_c2w)
            extri_opencv = pose_w2c[:3, :]  # (3, 4) [R|t]
            
            # 复制内参矩阵
            intri_opencv = intrinsic_matrix.copy()

            # 使用基类方法处理图像（包括调整大小、数据增强等）
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
                filepath=f"{seq_name}/frame_{image_idx:06d}",
            )

            # 验证输出尺寸
            if (image.shape[:2] != target_image_shape).any():
                logging.error(
                    f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}"
                )
                continue

            # 添加到输出列表
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # 构建输出batch
        set_name = "scanNetv2"
        batch = {
            "seq_name": f"{set_name}_{seq_name}",
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
    # 测试数据加载
    from hydra import initialize, compose
    
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
    
    # ⚠️ 路径修改：指向预处理后的数据目录
    dataset = ScanNetv2(
        common_conf=cfg.data.train.common_config,
        split="train",
        ScanNetv2_DIR="/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2_Processed",  # <- 修改此路径
        min_num_images=10,
    )
    
    print(f"Dataset initialized with {len(dataset.sequence_list)} scenes")
    
    # 测试数据获取
    if len(dataset.sequence_list) > 0:
        batch = dataset.get_data(seq_index=0, img_per_seq=5, aspect_ratio=1.0)
        print(f"Loaded batch: {batch['seq_name']}")
        print(f"  Frames: {batch['frame_num']}")
        print(f"  Image shape: {batch['images'][0].shape}")
        print(f"  Depth shape: {batch['depths'][0].shape}")
        print(f"  Intrinsics shape: {batch['intrinsics'][0].shape}")
        print(f"  Extrinsics shape: {batch['extrinsics'][0].shape}")
