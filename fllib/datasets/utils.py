import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from pycocotools import mask as coco_mask
from torchvision.transforms import functional as F
import json
import time
import shutil
import os
import csv
from collections import defaultdict
from pathlib import Path
from torchvision.datasets import GTSRB
from typing import Optional, Callable, Tuple, Any

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.images = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, label = self.images[index], self.labels[index]
        if self.transform_x is not None:
            data = self.transform_x(Image.open(data))
        else:
            data = Image.open(data)
        if self.transform_y is not None:
            label = self.transform_y(label)
        return data, label


class TransformDataset(Dataset):
    def __init__(self, images, labels, transform_x=None, transform_y=None):
        self.samples = images
        self.labels = labels
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.samples[idx]
        label = self.labels[idx]

        if self.transform_x:
            data = self.transform_x(data)
        if self.transform_y:
            label = self.transform_y(label)

        return data, label    


def collate_fn(batch):
    return tuple(zip(*batch))


def kitti_collate(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images, labels = zip(*data)

    imgs = []
    for img in images:
        img = np.array(img)
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = F.crop(img,0,0,370,1240)
        imgs.append(img)

    images = torch.stack(imgs,dim=0)

    return images,labels


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
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

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
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target    


class MyToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class MyCocoTransform(object):
    def __call__(self, image, target):

        anno = target["annotations"]
        bboxes = []
        ids = []
        target = {}
        for label in anno:
            bboxes.append([label['bbox'][0], 
                            label['bbox'][1],
                            label['bbox'][0] + label['bbox'][2],
                            label['bbox'][1] + label['bbox'][3]])
            ids.append(label['category_id'])

        target['boxes'] = torch.tensor(bboxes, dtype=torch.float)
        target['labels'] = torch.tensor(ids, dtype=torch.int64)


        return image, target
              
class MyGTSRBTargetTransform(object):
    def __call__(self, target):
        
        boxes = []
        labels = []
        targets = {}

        for tg in target:
            boxes.append([tg['Roi.X1'], tg['Roi.Y1'], tg['Roi.X2'], tg['Roi.Y2']])
            labels.append(tg['ClassId'])

        targets['boxes'] = torch.tensor(boxes, dtype=torch.float)
        targets['labels'] = torch.tensor(labels, dtype=torch.int64)

        return targets




class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform):
        super(CocoDetection, self).__init__(root, annFile)
        self._transforms = transform

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def processing_coco_input(images, labels):

    return images, labels


class BuildMiniCOCO:
    def __init__(self, annotation_file=None, origin_img_dir=""):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.origin_dir = origin_img_dir
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()  # imgToAnns　一个图片对应多个注解(mask) 一个类别对应多个图片
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index　　  给图片->注解,类别->图片建立索引
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def build(self, tarDir=None, tarFile='./new.json', N=1000):

        load_json = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances', "info": {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.", "url": "http:\/\/mscoco.org", "version": "1.0", "year": 2014, "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"}, "licenses": [{"url": "http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc\/2.0\/", "id": 2, "name": "Attribution-NonCommercial License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc-nd\/2.0\/",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by\/2.0\/", "id": 4, "name": "Attribution License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-sa\/2.0\/", "id": 5, "name": "Attribution-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nd\/2.0\/", "id": 6, "name": "Attribution-NoDerivs License"}, {"url": "http:\/\/flickr.com\/commons\/usage\/", "id": 7, "name": "No known copyright restrictions"}, {"url": "http:\/\/www.usa.gov\/copyright.shtml", "id": 8, "name": "United States Government Work"}]}
        if not Path(tarDir).exists():
            Path(tarDir).mkdir()

        for i in self.imgs:
            if(N == 0):
                break
            tic = time.time()
            img = self.imgs[i]
            load_json['images'].append(img)
            fname = os.path.join(tarDir, img['file_name'])
            anns = self.imgToAnns[img['id']]
            for ann in anns:
                load_json['annotations'].append(ann)
            if not os.path.exists(fname):
                shutil.copy(self.origin_dir+'/'+img['file_name'], tarDir)
            print('copy {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))
            N -= 1
        for i in self.cats:
            load_json['categories'].append(self.cats[i])
        with open(tarFile, 'w+') as f:
            json.dump(load_json, f, indent=4)


def coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
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

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset



class MyGTSRB(GTSRB):
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, transform, target_transform, download)

        if self._split == 'train':
            samples = self.read_dataset_from_csv()
        else:
            samples = self.read_dataset_from_csv()

        self._samples = samples
    
    def read_dataset_from_csv(self):
        instances = []
        directory = os.path.expanduser(self._target_folder)

        if self._split == 'train':
  
            classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())


            for class_dir in classes:
                target_dir = os.path.join(directory, class_dir)
                class_csv_file_name = os.path.join(target_dir, f'GT-{class_dir}.csv')
                
                with open(class_csv_file_name, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';', skipinitialspace=True)
                    for row in reader:
                        path = os.path.join(target_dir, row['Filename'])
                        item = path, [{k:float(v) for k, v in row.items() if k != 'Filename'}]                   
                        instances.append(item)
        else:
            test_csv_file = os.path.join(self._base_folder, 'GT-final_test.csv')

            with open(test_csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';', skipinitialspace=True)
                for row in reader:
                    path = os.path.join(directory, row['Filename'])
                    item = path, [{k:float(v) for k, v in row.items() if k != 'Filename'}]  
                    instances.append(item)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, targets = self._samples[index]

        sample = Image.open(path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return sample, targets


