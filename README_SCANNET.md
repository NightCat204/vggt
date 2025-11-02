# ScanNet 预处理与可视化（基于 vggt/training）

[toc]



## 数据准备

- 原始 ScanNetv2 目录应包含若干场景子目录及 `.sens` 文件，例如：
  - `ScanNetv2/scene0000_00/scene0000_00.sens`
- 示例数据已在本仓库 `ScanNetv2/scene0000_00/` 提供一个场景可快速试跑。

## 预处理（.sens → RGB/Depth/Pose 分离）

脚本位置：`vggt/training/data/preprocess/`

- 一键脚本（默认路径可覆盖）：
  ```bash
  cd vggt/training/data/preprocess
  bash scannet_preprocess.sh <输入原始目录> <输出目录>
  # 例：
  # bash scannet_preprocess.sh \
  #   /home/USER/datasets/ScanNetv2 \
  #   /home/USER/datasets/ScanNetv2_Processed
  ```
- 直接调用 Python（可加过滤/限帧）：
  ```bash
  cd vggt/training/data/preprocess
  python scannet_preprocess.py \
    --input_dir /path/to/ScanNetv2 \
    --output_dir /path/to/ScanNetv2_Processed \
    --jpeg_quality 95 \
    --scene_filter "scene0000*" \
    --max_frames 200 \
    --verbose
  ```
- 期望输出结构（每个场景目录）：
  ```
  ScanNetv2_Processed/
    scene0000_00/
      color/000000.jpg ...
      depth/000000.png  (uint16, 毫米)
      pose/000000.txt   (4x4 相机到世界)
      intrinsic.txt     (3x3 内参)
      metadata.txt
  ```

## 预处理结果验证（可选但推荐）

脚本：`vggt/training/test_scannet_preprocessed.py`

```bash
cd vggt/training
python test_scannet_preprocessed.py
# 如路径不同，请先在脚本内修改 SCANNET_PROCESSED_DIR
```

输出将打印帧数、图像/深度形状、深度统计等，以快速确认数据可读且数值合理。

## 可视化（基于 visualization/vis_dataset.py + Viser）

脚本：`vggt/training/visualization/vis_dataset.py`

- 启动（默认端口 8085）：
  ```bash
  cd vggt/training/visualization
  python vis_dataset.py --dataset scanNetv2 --seq_index 0 --img_per_seq 10
  ```
  打开浏览器访问 `http://localhost:8085`（脚本会打印一个可分享的临时 URL）。

- 常用参数：
  - `--seq_index`：场景索引（从 0 开始）
  - `--img_per_seq`：每次加载的帧数
  - `--point_size`：点大小（默认 0.001）
  - `--port`：Viser 端口（默认 8085）
  - `--no_depth`：关闭点云（仅相机视锥）
  - `--no_frustum`：关闭相机视锥
  - `--frustum_scale`：视锥显示尺度

- 运行逻辑要点：
  - 使用 Hydra 加载 `../config/default.yaml`（请在 `vggt/training/visualization` 目录下运行以保持相对路径正确）。
  - ScanNet 数据根路径来自 `vis_dataset.py` 内的 `PATH_ROOT_DICT['scanNetv2']`。
  - 数据加载由 `data/datasets/scanNetv2.py` 完成：读取 `color/*.jpg`、`depth/*.png`（毫米→米转换）、`pose/*.txt`（c2w → w2c）。

## 迁移到其他设备时需修改的位置

将路径切换到你的机器后，需要同步修改以下常量/默认值（按影响程度排序）：

1. 可视化入口路径（建议修改）
   - 文件：`vggt/training/visualization/vis_dataset.py`
   - 键：`PATH_ROOT_DICT['scanNetv2']`
   - 作用：指定预处理后的 ScanNet 根目录（传给 `ScanNetv2(ScanNetv2_DIR=...)`）。

2. 预处理默认输入/输出（可选）
   - 文件：`vggt/training/data/preprocess/scannet_preprocess.sh`
   - 变量：`INPUT_DIR`、`OUTPUT_DIR`
   - 说明：若直接用 Bash 脚本且不传参，需要把这两个默认路径改成你的本地目录。

3. 测试脚本中的硬编码路径（可选）
   - 文件：`vggt/training/test_scannet_preprocessed.py`
   - 常量：`SCANNET_PROCESSED_DIR`
   - 文件：`vggt/training/test_scannet_vis.py`
   - 常量：`SCANNET_DIR`

4. 数据集类默认参数（一般不必改）
   - 文件：`vggt/training/data/datasets/scanNetv2.py`
   - 参数：`ScanNetv2_DIR` 的默认值仅在未显式传入时才使用；本工程入口已显式传入，可忽略。

5. 端口与远程访问（可选）
   - `vis_dataset.py` 的 `--port`（默认 8085）。云/服务器部署需在安全范围内放通对应端口，或使用脚本打印的分享 URL。

6. 依赖环境（必要）
   - 至少安装：`viser`、`hydra-core`、`opencv-python`、`numpy`、`tqdm`。
   - 若你已按 `vggt/requirements_demo.txt` 安装，一般无需额外操作。

## 快速指令清单

```bash
# 1) 预处理（推荐）
cd vggt/training/data/preprocess
bash scannet_preprocess.sh /path/to/ScanNetv2 /path/to/ScanNetv2_Processed

# 2) 验证预处理结果（可选）
cd ../../
python test_scannet_preprocessed.py   # 若需，先改脚本内路径

# 3) 可视化（确保 PATH_ROOT_DICT['scanNetv2'] 已指向 _Processed 路径）
cd visualization
python vis_dataset.py --dataset scanNetv2 --seq_index 0 --img_per_seq 10
```

