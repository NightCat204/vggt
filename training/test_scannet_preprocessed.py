#!/usr/bin/env python3
"""
ScanNet预处理数据加载测试脚本

用于验证预处理后的ScanNet数据能否正确加载
"""

import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets.scanNetv2 import ScanNetv2
from hydra import initialize, compose
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_scannet_loading():
    """测试ScanNet数据集加载"""
    
    print("="*60)
    print("ScanNet预处理数据加载测试")
    print("="*60)
    
    # 加载配置（修正：config 在 training/config，不是上级目录）
    with initialize(version_base=None, config_path="./config"):
        cfg = compose(config_name="default")
    
    # 初始化数据集
    # ⚠️ 路径修改：指向预处理后的数据目录
    SCANNET_PROCESSED_DIR = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2_Processed"
    
    print(f"\n数据目录: {SCANNET_PROCESSED_DIR}")
    
    try:
        dataset = ScanNetv2(
            common_conf=cfg.data.train.common_config,
            split="train",
            ScanNetv2_DIR=SCANNET_PROCESSED_DIR,
            min_num_images=10,
        )
        
        print(f"✓ 数据集初始化成功")
        print(f"  场景数量: {len(dataset.sequence_list)}")
        
        if len(dataset.sequence_list) == 0:
            print("\n⚠️  警告: 未找到场景数据")
            print(f"  请确认以下路径存在预处理数据:")
            print(f"  {SCANNET_PROCESSED_DIR}")
            print(f"\n  运行预处理命令:")
            print(f"  cd training/data/preprocess")
            print(f"  bash scannet_preprocess.sh <输入目录> {SCANNET_PROCESSED_DIR}")
            return False
        
        # 测试数据加载
        print("\n开始测试数据加载...")
        batch = dataset.get_data(seq_index=0, img_per_seq=5, aspect_ratio=1.0)
        
        print(f"\n✓ 数据加载成功")
        print(f"  场景名称: {batch['seq_name']}")
        print(f"  帧数: {batch['frame_num']}")
        print(f"  图像尺寸: {batch['images'][0].shape}")
        print(f"  深度尺寸: {batch['depths'][0].shape}")
        print(f"  内参矩阵: {batch['intrinsics'][0].shape}")
        print(f"  外参矩阵: {batch['extrinsics'][0].shape}")
        
        # 数据有效性检查
        print("\n数据有效性检查:")
        import numpy as np
        
        # 检查深度值范围
        depth_sample = batch['depths'][0]
        depth_valid = depth_sample[depth_sample > 0]
        if len(depth_valid) > 0:
            print(f"  深度范围: {depth_valid.min():.3f}m ~ {depth_valid.max():.3f}m")
            print(f"  深度均值: {depth_valid.mean():.3f}m")
        else:
            print(f"  ⚠️  警告: 深度图无有效值")
        
        # 检查图像值范围
        img_sample = batch['images'][0]
        print(f"  图像范围: {img_sample.min()} ~ {img_sample.max()}")
        
        # 检查位姿矩阵
        extri = batch['extrinsics'][0]
        print(f"  位姿矩阵形状: {extri.shape} (应为 3x4)")
        
        print("\n" + "="*60)
        print("✓ 所有测试通过!")
        print("="*60)
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ 错误: 文件未找到")
        print(f"  {e}")
        print(f"\n请确认:")
        print(f"  1. 已运行预处理脚本")
        print(f"  2. 路径正确: {SCANNET_PROCESSED_DIR}")
        return False
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_scannet_loading()
    sys.exit(0 if success else 1)
