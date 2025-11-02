#!/bin/bash
# ScanNet数据预处理脚本
# 用法: bash scannet_preprocess.sh <input_dir> <output_dir>

# 默认路径
INPUT_DIR="${1:-/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2}"
OUTPUT_DIR="${2:-/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2_Processed}"

echo "=========================================="
echo "ScanNet数据预处理"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 运行预处理脚本
python scannet_preprocess.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --jpeg_quality 95 \
    --verbose

echo "=========================================="
echo "预处理完成！"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
