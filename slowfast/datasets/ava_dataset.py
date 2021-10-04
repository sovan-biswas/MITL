#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch
import random

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.AVA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.AVA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        self._load_data(cfg)
        self._neg_instant(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # self._image_paths, self._video_idx_to_name = ava_helper.load_clip_lists(
        #     cfg, is_train=(self._split == "train")
        # )

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        # load scene cut if training
        self._scene_cut = ava_helper.load_scene( cfg, self._video_idx_to_name, mode=self._split)

        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # # Get indices of keyframes and corresponding boxes and labels.
        # (
        #     self._keyframe_indices,
        #     self._keyframe_boxes_and_labels,
        # ) = ava_helper.get_keyvideo_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def _area(self, boxes):
        """Computes area of boxes.

      Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

      Returns:
        a numpy array with shape [N*1] representing box areas
      """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


    def _intersection(self, boxes1, boxes2):
        """Compute pairwise intersection areas between boxes.

      Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

      Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
      """
        [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
        [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

        all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
        all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
        intersect_heights = np.maximum(
            np.zeros(all_pairs_max_ymin.shape),
            all_pairs_min_ymax - all_pairs_max_ymin,
        )
        all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
        all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
        intersect_widths = np.maximum(
            np.zeros(all_pairs_max_xmin.shape),
            all_pairs_min_xmax - all_pairs_max_xmin,
        )
        return intersect_heights * intersect_widths

    def _iou(self, boxes1, boxes2):
        """Computes pairwise intersection-over-union between box collections.

      Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

      Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
      """
        intersect = self._intersection(boxes1, boxes2)
        area1 = self._area(boxes1)
        area2 = self._area(boxes2)
        union = (
            np.expand_dims(area1, axis=1)
            + np.expand_dims(area2, axis=0)
            - intersect
        )
        intersect[np.isnan( intersect )] = 0.000001
        union[np.isnan( union )] = 0.000001
        union[np.where(union == 0)] = 0.000001
        return intersect / union

    def _neg_peturb_box(self, boxes):
        for n in range(self.cfg.AVA.PERTURBATION):
            tmpbox = boxes[0:1,:]
            while (self._iou(tmpbox,boxes).max(1)>0.2):
                tmpbox = np.array([[random.random()/2.0, random.random()/2.0, 0.5+random.random()/2.0, 0.5+random.random()/2.0]])
            boxes = np.concatenate((boxes,tmpbox),axis=0)
        return boxes


    def _processitem__(self, idx, original_flag=0):
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        scores = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
            scores.append(box_labels[2])
        boxes = np.array(boxes)

        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()
        scores = np.array(scores)



        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )
        # image_paths = self._image_paths[video_idx][round(center_idx)]
        # img_sec = int(image_paths.split('/')[-1].split('.')[0])
        # assert (img_sec == sec)
        # # print(image_paths,sec_idx,sec,img_sec==sec)
        # imgs = utils.retry_load_video(
        #     image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        # )

        ## Get Negative Perturbation box
        boxes = self._neg_peturb_box(boxes) if self.cfg.TRAIN.ENABLE and self._split == 'train' else boxes
        # mid_fr = int(len(imgs) / 2)
        # if self.cfg.TRAIN.ENABLE and (self._split == "train"):
        #     val = random.uniform(0, 1)
        #     if val > 0.5:
        #         mid_fr = mid_fr - 1
        # seq = utils.get_sequence(
        #     mid_fr,
        #     self._seq_len // 2,
        #     self._sample_rate,
        #     num_frames=len(imgs),
        # )
        # imgs = [imgs[frame] for frame in seq]
        assert (len(imgs) == int(self._seq_len / self._sample_rate))
        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        metadata = [[video_idx, sec]] * len(ori_boxes)

        if original_flag != 0 and boxes.shape[0] > 5:
            randomize = np.arange(boxes.shape[0])
            np.random.shuffle(randomize)
            clip_locations = min(boxes.shape[0], 5)
            boxes = boxes[randomize][:clip_locations]
            ori_boxes = ori_boxes[randomize][:clip_locations]
            scores = scores[randomize][:clip_locations]
            label_arrs = label_arrs[randomize][:clip_locations]

        extra_data = {
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
            "scores": scores,
        }

        return imgs, label_arrs, idx, extra_data

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        if self._split == 'train' and self.cfg.MODEL.CONTRASTIVE:
            # neg_idx = self._get_contrast(idx)
            neg_idx = min(max(idx + random.choice(list(range(-800, -100))+list(range(101, 801))), 0), len(self._keyframe_indices) - 1)
            perturb = min(max(idx+random.choice(list([-1,0,1])),0),len(self._keyframe_indices)-1)
            video_idx1, sec_idx1, sec1, _ = self._keyframe_indices[idx]
            video_idx2, sec_idx2, sec2, _ = self._keyframe_indices[perturb]
            video_idx3, sec_idx3, sec3, _ = self._keyframe_indices[neg_idx]
            # label1 = set(sum([l[1] for l in self._keyframe_boxes_and_labels[video_idx1][sec_idx1]],[]))
            # label1.discard(-1)
            # label2 = set(sum([l[1] for l in self._keyframe_boxes_and_labels[video_idx2][sec_idx2]],[]))
            # label2.discard(-1)

            while video_idx1 != video_idx2 or abs(int(sec1) - int(sec2))> 1 or \
                    (self._scene_cut != None and (self._scene_cut[video_idx1][sec1-902] != self._scene_cut[video_idx2][sec2-902])):
                perturb = min(max(idx + random.choice(list([-1, 0, 1])), 0), len(self._keyframe_indices) - 1)
                video_idx2, sec_idx2, sec2, _ = self._keyframe_indices[perturb]
                # label2 = set(sum([l[1] for l in self._keyframe_boxes_and_labels[video_idx2][sec_idx2]],[]))
                # label2.discard(-1)

            while video_idx1 != video_idx3 or abs(int(sec1) - int(sec3))< 100:
                neg_idx = min(max(idx + random.choice(list(range(-800, -100))+list(range(101, 801))), 0), len(self._keyframe_indices) - 1)
                video_idx3, sec_idx3, sec3, _ = self._keyframe_indices[neg_idx]

            # idxs = [idx,min(max(idx+random.choice(list([-1,0,1])),0),len(self._keyframe_indices)-1),neg_idx]
            idxs = [idx, perturb, neg_idx]
            # idxs = [idx, idx, neg_idx]
        else:
            idxs = [idx]
        return [self._processitem__(idx, idx_num) for idx_num, idx in enumerate(idxs)]

    def _get_contrast(self, idx):
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        label_all = []
        for l in self._keyframe_boxes_and_labels[video_idx][sec_idx]:
            label_all.extend(l[1])
        label_all = set(label_all)
        label_all.discard(-1)
        label_all = list(label_all)
        random.shuffle(label_all)
        not_label_all = label_all[:int(0.6*len(label_all))]
        for l in label_all[int(0.6*len(label_all)):]:
            x = x.intersection(set(self.pos_dict[l])) if 'x' in locals() else set(self.pos_dict[l])
        for l in not_label_all:
            x = x.intersection(set(self.negative_dict[l])) if 'x' in locals() and len(x) > 0 else set(self.negative_dict[l])
        return random.choice(list(x)) if 'x' in locals() and len(x) > 0 else random.choice(self.negative_dict[random.randint(1,self.cfg.MODEL.NUM_CLASSES)])

    def _neg_instant(self,cfg):
        num_classes = set([i + 1 for i in range(cfg.MODEL.NUM_CLASSES)])
        self.negative_dict = dict([(i+1,[]) for i in range(cfg.MODEL.NUM_CLASSES)])
        self.pos_dict = dict([(i + 1, []) for i in range(cfg.MODEL.NUM_CLASSES)])
        num_indxs = len(self._keyframe_indices)
        for i in range(num_indxs):
            video_idx, sec_idx, sec, center_idx = self._keyframe_indices[i]
            label_all = []
            for l in self._keyframe_boxes_and_labels[video_idx][sec_idx]:
                label_all.extend(l[1])
            label_all = set(label_all)
            label_all.discard(-1)
            for j in label_all:
                self.pos_dict[j].append(i)
            for j in (num_classes ^ label_all):
                self.negative_dict[j].append(i)
        return
