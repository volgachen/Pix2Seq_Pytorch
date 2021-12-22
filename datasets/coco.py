# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_set, args=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.large_scale_jitter = args.large_scale_jitter
        self.image_set = image_set

        if image_set == "train" and args.remove_empty_annotations:
            self.ids = sorted(self.ids)
            ids = []
            obj_counts = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
                    obj_counts.append(len([obj for obj in anno if obj['iscrowd'] == 0 and obj["bbox"][2] > 0 and obj["bbox"][3] > 0]))
            if args.data_divide:
                indexs = torch.as_tensor(obj_counts).argsort()
                self.ids = [ids[i] for i in indexs]
                print("Rearrange Data Sequence...")
            else:
                self.ids = ids

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        filename = os.path.join(self.root, path)
        return Image.open(filename).convert("RGB")

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            if self.large_scale_jitter and self.image_set == "train":
                img1, target1 = self._transforms(img, target)
                img2, target2 = self._transforms(img, target)
                return img1, img2, target1, target2
            else:
                img, target = self._transforms(img, target)
                return img, target
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

            polygons = [torch.tensor(obj["segmentation"][0]) for obj in anno]
            num_per_polygon = torch.tensor([p.shape[0] for p in polygons], dtype=torch.int64)
            new_polygons = torch.zeros([len(polygons), max(num_per_polygon)])
            for gt_i, (np, p) in enumerate(zip(num_per_polygon, polygons)):
                new_polygons[gt_i, :np] = p
            target["polygons"] = new_polygons
            target["valid_pol_idx"] = num_per_polygon

        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        transforms = [T.RandomHorizontalFlip()]
        if args.large_scale_jitter and args.aug_scale_min < 1.0 and args.aug_scale_max > 1.0:
            transforms.append(T.LargeScaleJitter(output_size=args.max_input_size, aug_scale_min=args.aug_scale_min, aug_scale_max=args.aug_scale_max))
        else:
            transforms.append(T.RandomResize([args.max_input_size], max_size=args.max_input_size))
        if args.color_distortion:
            transforms.append(T.RandomDistortion(0.5, 0.5, 0.5, 0.5))
        transforms.extend([
            normalize,
            T.TargetPermute(),
            T.SeqBuilder(args),
        ])
        print(transforms)
        return T.Compose(transforms)

    if image_set == 'val':
        if args.large_scale_jitter:
            return T.Compose([
                T.LargeScaleJitter(output_size=args.max_input_size, aug_scale_min=1.0, aug_scale_max=1.0),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResize([800], max_size=args.max_input_size),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = args.coco_path
    mode = 'instances'
    root = Path(root)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=False,
        image_set=image_set,
        args=args)
    return dataset
