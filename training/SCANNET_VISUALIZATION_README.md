# ScanNetv2 可视化集成说明

## 概述

本次修改实现了在VGGT训练框架中直接从ScanNet `.sens` 原始文件读取数据并进行可视化，无需预处理成RGB/深度/位姿的独立文件。

---

## 核心修改

### 1. **新增文件**

#### `training/utils/scannet_sens_reader.py`

- 从 `scannet_visualization` 项目移植
- 实现 `.sens` 文件的二进制解析
- 核心原理：
  ```
  .sens 文件结构：
  ├─ Header (元数据)
  │  ├─ 相机内参 (color + depth, 4×4)
  │  ├─ 图像分辨率
  │  ├─ 深度单位 (depth_shift)
  │  └─ 总帧数
  └─ Frames (逐帧数据)
     ├─ 相机位姿 (4×4 camera-to-world)
     ├─ RGB图像 (JPEG压缩)
     └─ 深度图 (zlib压缩的uint16)
  ```

### 2. **修改文件**

#### `training/data/datasets/scanNetv2.py`

**关键改动**：

1. **导入sens读取器**：

   ```python
   from utils.scannet_sens_reader import load_sens_file
   ```
2. **初始化逻辑**：

   - 扫描 `*.sens` 文件而非目录结构
   - 缓存已加载的 `SensorData` 对象（避免重复IO）
   - 读取header验证帧数是否满足 `min_num_images`
3. **get_data方法**：

   - 从缓存的 `SensorData` 读取RGB/深度/位姿
   - 深度单位转换：`uint16 (mm) → float32 (m)`
   - 位姿转换：`camera-to-world → world-to-camera` (OpenCV约定)
   - 使用color相机内参（可配置）

#### `training/visualization/vis_dataset.py`

- 更新 `PATH_ROOT_DICT['scanNetv2']` 指向本地路径

# ScanNetv2 可视化集成说明

## 概述

本次修改实现了在VGGT训练框架中直接从ScanNet `.sens` 原始文件读取数据并进行可视化，无需预处理成RGB/深度/位姿的独立文件。

---

## 路径配置指南（移植）

在新环境部署时，**必须**修改以下文件中的路径配置：

### **配置清单（按优先级排序）**

| 文件                                      | 行号 | 变量/键名                       | 说明             |
| ----------------------------------------- | ---- | ------------------------------- | ---------------- |
| **1. visualization/vis_dataset.py** | ~212 | `PATH_ROOT_DICT['scanNetv2']` | 可视化入口路径   |
| **2. data/datasets/scanNetv2.py**   | ~38  | `ScanNetv2_DIR` 参数默认值    | 数据集类默认路径 |
| **3. test_scannet_vis.py**          | ~31  | `SCANNET_DIR` 变量            | 测试脚本路径     |

### **详细修改步骤**

#### **步骤1: 确定您的数据集位置**

```bash
# 找到您的ScanNetv2数据集目录
# 应该包含如下结构：
# ScanNetv2/
#   scene0000_00/
#     scene0000_00.sens
#   scene0001_00/
#     scene0001_00.sens
#   ...

# 示例路径（根据您的实际情况选择）:
# Linux:   /home/username/datasets/ScanNetv2
```

#### **步骤2: 修改vis_dataset.py（必需）**

打开 `visualization/vis_dataset.py`，找到第 ~212 行：

```python
# 当前路径（示例）
'scanNetv2': '/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2',
```

#### **步骤3: 修改scanNetv2.py（推荐）**

打开 `data/datasets/scanNetv2.py`，找到第 ~38 行：

```python
# 当前默认值（示例）
ScanNetv2_DIR: str = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2",

# 修改为您的路径
ScanNetv2_DIR: str = "/YOUR/ACTUAL/PATH/TO/ScanNetv2",  # <- 修改这里
```

#### **步骤4: 修改test_scannet_vis.py（推荐）**

打开 `test_scannet_vis.py`，找到第 ~31 行：

```python
# 当前路径（示例）
SCANNET_DIR = "/home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/ScanNetv2"

# 修改为您的路径
SCANNET_DIR = "/YOUR/ACTUAL/PATH/TO/ScanNetv2"  # <- 修改这里
```

---

## 使用方法

### 快速测试

```bash
cd /home/wpy/Personal/File/Research/LargeOdometryModel/VGGT_Test/vggt/training

# 1. 测试数据加载
python test_scannet_vis.py

# 2. 运行可视化
cd visualization
python vis_dataset.py \
    --dataset scanNetv2 \
    --seq_index 0 \
    --img_per_seq 10 \
    --point_size 0.003 \
    --port 8085 \
    --no_frustum
```

### 参数说明

| 参数                | 说明            | 默认值     |
| ------------------- | --------------- | ---------- |
| `--dataset`       | 数据集名称      | `vkitti` |
| `--seq_index`     | 场景索引        | `10`     |
| `--img_per_seq`   | 每场景帧数      | `24`     |
| `--point_size`    | 点云大小        | `0.001`  |
| `--port`          | Viser服务器端口 | `8085`   |
| `--frustum_scale` | 视锥体缩放比例  | `0.5`    |

**布尔参数用法**

- `--no_frustum` - 禁用相机视锥体显示（**不要**使用 `--show_frustum False`）
- `--no_depth` - 禁用深度可视化（**不要**使用 `--visualize_depth False`）

---

## 数据流程对比

### **原方案（需要预处理）**：

```
.sens文件 → 预处理脚本 → color/*.jpg + depth/*.png + pose/*.txt 
                           ↓
                   scanNetv2.py 读取预处理文件
```

### **新方案（直接读取）**：

```
.sens文件 → scannet_sens_reader.py 实时解析 
              ↓
          scanNetv2.py 直接获取数据
```

---

## 技术细节

### 坐标系转换

```python
# SENS文件存储: camera-to-world (pose_c2w)
# VGGT训练框架需要: world-to-camera (pose_w2c)

pose_c2w = sensor_data.camera_poses[i]  # (4, 4)
pose_w2c = np.linalg.inv(pose_c2w)      # 取逆
extri_opencv = pose_w2c[:3, :]           # (3, 4) [R|t]
```

### 深度处理

```python
# SENS: uint16 millimeters
depth_mm = sensor_data.depth_images[i]

# 转换为米并过滤
depth_m = depth_mm.astype(np.float32) / 1000.0
depth_m = threshold_depth_map(depth_m, max_depth=10.0)
```

### 内参选择

```python
# 可配置使用color或depth相机内参
if use_color_intrinsics:
    K = sensor_data.get_color_intrinsics_3x3()  # 1296×968
else:
    K = sensor_data.get_depth_intrinsics_3x3()  # 640×480
```
