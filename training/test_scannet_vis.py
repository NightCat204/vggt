#!/usr/bin/env python
"""
Test script for ScanNetv2 dataset visualization
Verifies that .sens file loading and visualization work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from hydra import initialize, compose
from data.datasets.scanNetv2 import ScanNetv2

def test_scannet_loading():
    """Test loading ScanNetv2 data from .sens files"""
    
    print("="*60)
    print("Testing ScanNetv2 Dataset Loading")
    print("="*60)
    
    # Initialize config
    with initialize(version_base=None, config_path="./config"):
        cfg = compose(config_name="default")
    
    # ⚠️ PATH TO MODIFY: Update ScanNetv2_DIR to your dataset location
    # This should be the root directory containing scene subdirectories with .sens files
    # Examples:
    #   Linux:   "/home/username/datasets/ScanNetv2"
    #   macOS:   "/Users/username/datasets/ScanNetv2"
    #   Windows: "C:/datasets/ScanNetv2" or "D:/datasets/ScanNetv2"
    SCANNET_DIR = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2"  # <- CHANGE THIS
    
    # Create dataset
    print("\n1. Initializing ScanNetv2 dataset...")
    print(f"   Path: {SCANNET_DIR}")
    dataset = ScanNetv2(
        common_conf=cfg.data.train.common_config,
        split="train",
        ScanNetv2_DIR=SCANNET_DIR,
        min_num_images=5,
        max_frames_per_scene=100,  # Limit for faster testing
        use_color_intrinsics=True,
    )
    
    print(f"   ✓ Found {dataset.sequence_list_len} valid scenes")
    
    if dataset.sequence_list_len == 0:
        print("   ✗ No scenes found! Check your ScanNetv2_DIR path.")
        return False
    
    # Test data loading
    print("\n2. Loading sample batch...")
    try:
        batch = dataset.get_data(seq_index=0, img_per_seq=3, aspect_ratio=1.0)
        
        print(f"   ✓ Loaded scene: {batch['seq_name']}")
        print(f"   ✓ Frames loaded: {batch['frame_num']}")
        
        if batch['frame_num'] > 0:
            print(f"\n3. Batch contents:")
            print(f"   - Image shape: {batch['images'][0].shape}")
            print(f"   - Depth shape: {batch['depths'][0].shape}")
            print(f"   - Intrinsics shape: {batch['intrinsics'][0].shape}")
            print(f"   - Extrinsics shape: {batch['extrinsics'][0].shape}")
            print(f"   - World points shape: {batch['world_points'][0].shape}")
            
            # Check data validity
            import numpy as np
            img = batch['images'][0]
            depth = batch['depths'][0]
            
            print(f"\n4. Data validation:")
            print(f"   - Image range: [{img.min()}, {img.max()}]")
            print(f"   - Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
            print(f"   - Valid depth ratio: {np.sum(depth > 0) / depth.size:.2%}")
            
            print("\n" + "="*60)
            print("✓ All tests passed!")
            print("="*60)
            return True
        else:
            print("   ✗ No frames loaded")
            return False
            
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scannet_loading()
    
    if success:
        print("\n" + "="*60)
        print("✓ Tests passed! Next step: Run visualization")
        print("="*60)
        print("\nCommand:")
        print("  cd visualization")
        print("  python vis_dataset.py --dataset scanNetv2 --seq_index 0 --img_per_seq 10")
        print("\nOr use the quick launch script:")
        print("  ./run_scannet_vis.sh 0 10")
        print("\nThis will:")
        print("  - Load scene from .sens file")
        print("  - Generate 3D point clouds")
        print("  - Display in Viser (http://localhost:8085)")
        print("\n⚠️  IMPORTANT: Before running, verify all paths are correctly set!")
        print("   See SCANNET_VISUALIZATION_README.md for details.")
    else:
        print("\n✗ Tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Verify ScanNetv2_DIR path exists")
        print("  2. Check .sens files are present: ls <ScanNetv2_DIR>/*/*.sens")
        print("  3. Review path configuration in:")
        print("     - test_scannet_vis.py (line ~29)")
        print("     - visualization/vis_dataset.py (line ~217)")
        print("     - data/datasets/scanNetv2.py (line ~40)")
    
    sys.exit(0 if success else 1)
