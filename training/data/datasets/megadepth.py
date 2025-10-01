# data/datasets/megadepth.py
import os
import os.path as osp
import logging
import random
import glob
import h5py
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset

from hydra import initialize, compose


class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MEGADEPTH_DIR: str = "MegaDepth/phoenix/S6/zl548/MegaDepth_v1",
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.MEGADEPTH_DIR = MEGADEPTH_DIR
        self.min_num_images = int(min_num_images)

        if split == "train":
            self.len_train = int(len_train)
        elif split == "test":
            self.len_train = int(len_test)
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"MEGADEPTH_DIR is {self.MEGADEPTH_DIR}")

        sequence_list, sequence_lengths = [], []
        scenes = sorted([d for d in os.listdir(self.MEGADEPTH_DIR) if os.path.isdir(osp.join(self.MEGADEPTH_DIR, d))])
        for scene in scenes:
            scene_dir = osp.join(self.MEGADEPTH_DIR, scene)
            dense_dirs = sorted([d for d in os.listdir(scene_dir) if d.startswith("dense") and os.path.isdir(osp.join(scene_dir, d))])
            for dense_name in dense_dirs:
                dense_dir  = osp.join(scene_dir, dense_name)
                imgs_dir   = osp.join(dense_dir, "imgs")
                depths_dir = osp.join(dense_dir, "depths")
                poses_dir  = osp.join(dense_dir, "camera_pose")
                if not (osp.isdir(imgs_dir) and osp.isdir(depths_dir) and osp.isdir(poses_dir)):
                    continue

                depth_files = sorted(glob.glob(osp.join(depths_dir, "**", "*.h5"), recursive=True))
                count = 0
                for dp in depth_files:
                    rel = osp.relpath(dp, depths_dir)
                    img_jpg = osp.join(imgs_dir,   osp.splitext(rel)[0] + ".jpg")
                    img_png = osp.join(imgs_dir,   osp.splitext(rel)[0] + ".png")
                    pose_npz= osp.join(poses_dir,  osp.splitext(rel)[0] + ".npz")
                    if osp.isfile(pose_npz) and (osp.isfile(img_jpg) or osp.isfile(img_png)):
                        count += 1
                if count >= self.min_num_images:
                    sequence_list.append((scene, dense_name))
                    sequence_lengths.append(count)

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.save_seq_stats(self.sequence_list, sequence_lengths, self.__class__.__name__)

        self.depth_max = 80.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MegaDepth scene count: {self.sequence_list_len}")
        logging.info(f"{status}: MegaDepth dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            scene, dense_name = self.sequence_list[seq_index]
        else:
            scene, dense_name = seq_name.split("/")

        dense_dir  = osp.join(self.MEGADEPTH_DIR, scene, dense_name)
        imgs_dir   = osp.join(dense_dir, "imgs")
        depths_dir = osp.join(dense_dir, "depths")
        poses_dir  = osp.join(dense_dir, "camera_pose")

        depth_files = sorted(glob.glob(osp.join(depths_dir, "**", "*.h5"), recursive=True))
        tags = []
        for dp in depth_files:
            rel = osp.relpath(dp, depths_dir)
            img_jpg = osp.join(imgs_dir,   osp.splitext(rel)[0] + ".jpg")
            img_png = osp.join(imgs_dir,   osp.splitext(rel)[0] + ".png")
            pose_npz= osp.join(poses_dir,  osp.splitext(rel)[0] + ".npz")
            if osp.isfile(pose_npz) and (osp.isfile(img_jpg) or osp.isfile(img_png)):
                tags.append(rel)

        num_images = len(tags)
        if num_images < 1:
            raise ValueError(f"No valid frames in {seq_name}")

        if img_per_seq is None:
            img_per_seq = min(24, num_images)
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths = [], []
        cam_points, world_points, point_masks = [], [], []
        extrinsics, intrinsics, original_sizes = [], [], []

        for local_idx in ids:
            rel = tags[int(local_idx)]
            img_path = osp.join(imgs_dir,   osp.splitext(rel)[0] + ".jpg")
            if not osp.isfile(img_path):
                img_path = osp.join(imgs_dir, osp.splitext(rel)[0] + ".png")
            depth_path = osp.join(depths_dir, rel)
            pose_path  = osp.join(poses_dir,  osp.splitext(rel)[0] + ".npz")

            image = read_image_cv2(img_path)

            with h5py.File(depth_path, "r") as hf:
                depth_map = np.asarray(hf["depth"], dtype=np.float32)

            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Shape mismatch: {image.shape[:2]} vs {depth_map.shape}"
            original_size = np.array(image.shape[:2], dtype=np.int32)

            data = np.load(pose_path)
            intri_opencv = data["K"].astype(np.float32).reshape(3,3)
            R_wc = data["R"].astype(np.float32).reshape(3,3)
            t_wc = data["T"].astype(np.float32).reshape(3)
            extri_opencv = np.concatenate([R_wc, t_wc.reshape(3,1)], axis=1).astype(np.float32)

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
                filepath=img_path,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "megadepth"
        batch = {
            "seq_name": f"{set_name}_{scene}_{dense_name}",
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
    ds = MegaDepthDataset(
        common_conf=cfg.data.train.common_config,
        split="train",
        MEGADEPTH_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_av_verifvalid/users/ymingli/datasets/MegaDepth/phoenix/S6/zl548/MegaDepth_v1",
    )
    print(ds[(0, 24, 1.0)])
