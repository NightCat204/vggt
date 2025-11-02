#!/usr/bin/env python3
"""
ScanNet数据预处理脚本

功能：从.sens文件提取并保存RGB/depth/pose到独立文件
目标结构：
    ScanNetv2_Processed/
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
            intrinsic.txt  (3x3 camera intrinsics)
            metadata.txt   (scene metadata)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import glob
from tqdm import tqdm

import cv2
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.scannet_sens_reader import load_sens_file


def setup_logging(verbose: bool = False):
    """配置日志输出"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )


def preprocess_scene(
    sens_path: str,
    output_dir: str,
    max_frames: Optional[int] = None,
    use_color_intrinsics: bool = True,
    jpeg_quality: int = 95
) -> dict:
    """
    预处理单个场景
    
    Args:
        sens_path: .sens文件路径
        output_dir: 输出目录
        max_frames: 最大帧数限制
        use_color_intrinsics: 是否使用彩色相机内参（否则使用深度相机）
        jpeg_quality: JPEG压缩质量 (0-100)
        
    Returns:
        处理统计信息字典
    """
    scene_name = Path(sens_path).parent.name
    scene_output = Path(output_dir) / scene_name
    
    logging.info(f"Processing scene: {scene_name}")
    
    # 创建输出目录
    (scene_output / "color").mkdir(parents=True, exist_ok=True)
    (scene_output / "depth").mkdir(parents=True, exist_ok=True)
    (scene_output / "pose").mkdir(parents=True, exist_ok=True)
    
    # 加载.sens文件
    try:
        sensor_data = load_sens_file(sens_path, max_frames=max_frames)
    except Exception as e:
        logging.error(f"Failed to load {sens_path}: {e}")
        return {"success": False, "error": str(e)}
    
    num_frames = len(sensor_data.color_images)
    logging.info(f"  Loaded {num_frames} frames from .sens file")
    
    # 保存内参矩阵
    if use_color_intrinsics:
        intrinsics = sensor_data.get_color_intrinsics_3x3()
        logging.info(f"  Using color intrinsics: {sensor_data.color_width}x{sensor_data.color_height}")
    else:
        intrinsics = sensor_data.get_depth_intrinsics_3x3()
        logging.info(f"  Using depth intrinsics: {sensor_data.depth_width}x{sensor_data.depth_height}")
    
    np.savetxt(scene_output / "intrinsic.txt", intrinsics, fmt='%.6f')
    
    # 保存元数据
    metadata = {
        "scene_name": scene_name,
        "num_frames": num_frames,
        "color_width": sensor_data.color_width,
        "color_height": sensor_data.color_height,
        "depth_width": sensor_data.depth_width,
        "depth_height": sensor_data.depth_height,
        "depth_shift": sensor_data.depth_shift,
        "intrinsics_type": "color" if use_color_intrinsics else "depth"
    }
    
    with open(scene_output / "metadata.txt", 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    # 逐帧保存数据
    success_count = 0
    for i in tqdm(range(num_frames), desc=f"  Saving frames", leave=False):
        try:
            color_img, depth_img, pose = sensor_data.get_frame_data(i)
            
            if color_img is None or depth_img is None:
                logging.warning(f"  Skipping frame {i} (invalid data)")
                continue
            
            # 保存RGB图像 (JPEG格式)
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(scene_output / "color" / f"{i:06d}.jpg"),
                color_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            # 保存深度图像 (PNG格式, uint16, millimeters)
            cv2.imwrite(
                str(scene_output / "depth" / f"{i:06d}.png"),
                depth_img
            )
            
            # 保存位姿矩阵 (4x4 camera-to-world)
            np.savetxt(
                scene_output / "pose" / f"{i:06d}.txt",
                pose,
                fmt='%.6f'
            )
            
            success_count += 1
            
        except Exception as e:
            logging.warning(f"  Failed to save frame {i}: {e}")
            continue
    
    stats = {
        "success": True,
        "scene_name": scene_name,
        "total_frames": num_frames,
        "saved_frames": success_count,
        "color_resolution": f"{sensor_data.color_width}x{sensor_data.color_height}",
        "depth_resolution": f"{sensor_data.depth_width}x{sensor_data.depth_height}"
    }
    
    logging.info(f"  ✓ Saved {success_count}/{num_frames} frames")
    return stats


def main():
    parser = argparse.ArgumentParser(description="ScanNet数据预处理工具")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="ScanNet原始数据目录（包含.sens文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="预处理输出目录"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="每个场景最大帧数限制（默认：全部）"
    )
    parser.add_argument(
        "--use_depth_intrinsics",
        action="store_true",
        help="使用深度相机内参（默认：彩色相机）"
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG压缩质量 (0-100, 默认：95)"
    )
    parser.add_argument(
        "--scene_filter",
        type=str,
        default=None,
        help="场景名过滤器（例如：scene0000*）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # 查找所有.sens文件
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return
    
    if args.scene_filter:
        pattern = f"{args.scene_filter}/*.sens"
    else:
        pattern = "*/*.sens"
    
    sens_files = sorted(glob.glob(str(input_dir / pattern)))
    
    if not sens_files:
        logging.error(f"No .sens files found in {input_dir}")
        return
    
    logging.info(f"Found {len(sens_files)} scenes to process")
    logging.info(f"Output directory: {args.output_dir}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理所有场景
    all_stats = []
    for sens_path in tqdm(sens_files, desc="Processing scenes"):
        stats = preprocess_scene(
            sens_path=sens_path,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            use_color_intrinsics=not args.use_depth_intrinsics,
            jpeg_quality=args.jpeg_quality
        )
        all_stats.append(stats)
    
    # 输出总结
    success_scenes = [s for s in all_stats if s.get("success", False)]
    total_frames = sum(s.get("saved_frames", 0) for s in success_scenes)
    
    logging.info("\n" + "="*60)
    logging.info("预处理完成")
    logging.info(f"成功处理场景: {len(success_scenes)}/{len(all_stats)}")
    logging.info(f"总帧数: {total_frames}")
    logging.info("="*60)
    
    # 保存处理统计
    stats_file = Path(args.output_dir) / "preprocessing_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Total scenes: {len(all_stats)}\n")
        f.write(f"Success: {len(success_scenes)}\n")
        f.write(f"Total frames: {total_frames}\n\n")
        for s in success_scenes:
            f.write(f"{s['scene_name']}: {s['saved_frames']} frames\n")
    
    logging.info(f"统计信息已保存至: {stats_file}")


if __name__ == "__main__":
    main()
