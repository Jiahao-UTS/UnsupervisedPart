import cv2
import json
import numpy as np
import os
import utils
from PIL import Image
import pandas as pd
import skimage.draw
from pycocotools.coco import COCO

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

def transformation_from_points(src, target):
    src = np.matrix(src.astype(np.float64))
    target = np.matrix(target.astype(np.float64))

    center_src = np.mean(src, axis=0)
    center_target = np.mean(target, axis=0)

    src = src - center_src
    target = target - center_target

    std_src = np.std(src)
    std_target = np.std(target)

    src /= std_src
    target /= std_target

    U, S, Vt = np.linalg.svd(src.T * target)

    R = (U * Vt).T
    return np.vstack([np.hstack(((std_target/ std_src) * R,
                                 center_target.T-(std_target/std_src) * R * center_src.T)),
                      np.matrix([0., 0., 1.])])


def get_perspective_matrix(src, dst):
    new_src = []
    new_dst = []
    for point in src:
        new_src.append((point[0], point[1]))
    for point in dst:
        new_dst.append((point[0], point[1]))
    matrix = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32))
    return matrix[0]

def affine_matrix_invertible(affine_matrix):
    affine_matrix = np.concatenate((affine_matrix, np.array([[0.0, 0.0, 1.0]])), axis=0)
    return np.linalg.matrix_rank(affine_matrix) == affine_matrix.shape[0]

def affine_transform(point, affine_matrix):
    point_num = len(point)
    one_matrix = np.ones((point_num, 1))
    new_point = np.concatenate((point, one_matrix), axis=1).T
    new_point = np.dot(affine_matrix, new_point).T
    return new_point[:,:2] / new_point[:, 2:3]

def get_pair_transform(trans_mat1, trans_mat2):
    trans_mat1 = np.concatenate((trans_mat1, np.array([[0.0, 0.0, 1.0]])), axis=0)
    trans_mat2 = np.concatenate((trans_mat2, np.array([[0.0, 0.0, 1.0]])), axis=0)

    trans_mat1 = np.linalg.inv(trans_mat1)
    trans_pair = trans_mat2 @ trans_mat1

    return trans_pair[0:2, :]

def pil_loader(path, type):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(type)

class PartImage_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.PartImage.FRACTION
        self.Translation_Factor = cfg.PartImage.TRANSLATION
        self.Rotation_Factor = cfg.PartImage.ROTATION
        self.Scale_Factor = cfg.PartImage.SCALE
        self.Flip = cfg.PartImage.FLIP

        with open(os.path.join(root, "label_2_supercategory.json"), "r", encoding='utf-8') as f:
            self.super_category = json.load(f)
            f.close()

        self.Transform = transform

        Dataset = pd.read_csv(os.path.join(root, "newdset.txt"), sep='\t',
                              names=["index", "test", "label", "class", "filename"])

        if is_train:
            self.Dataset = Dataset.loc[(Dataset['test'] == 0)]
        else:
            self.Dataset = Dataset.loc[(Dataset['test'] == 1)]

        annFile = os.path.join(root, f"train.json")
        coco = COCO(annFile)
        self.coco = coco

    def getmasks(self, i):
        idx = self.Dataset.iloc[i]['index']
        idx = int(idx)
        coco = self.coco
        img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        cat_ids = [ann['category_id'] for ann in anns]
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(poly)
        for cat, p in zip(cat_ids, polygons):
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            try:
                mask_tensor[cat] += torch.FloatTensor(mask)
            except NameError:
                mask_tensor = torch.zeros(size=(40, mask.shape[-2], mask.shape[-1]))
                mask_tensor[cat] += torch.FloatTensor(mask)
        try:
            mask_tensor = torch.where(mask_tensor > 0.1, 1, 0).permute(0, 2, 1)
            return mask_tensor
        except UnboundLocalError:
            # if an image has no ground truth parts
            return None

        mask = self.getmasks(idx)
        if mask == None:
            mask = torch.zeros(size=(40, im.shape[-2], im.shape[-1]))
        mask = transforms.Resize(size=(im.shape[-2], im.shape[-1]),
                interpolation=transforms.InterpolationMode.NEAREST)(mask)
        return im, label, mask

    @staticmethod
    def only_file_names(lst):
        return [e['file_name'] for e in lst]

    def Image_Flip(self, Img, bbox):
        Img = cv2.flip(Img, 1)

        width = Img.shape[1]
        bbox[0] = width - 1 - bbox[2] - bbox[0]

        return Img, bbox

    def get_torch_theta(self, warp_matrix):
        warp_matrix = np.concatenate((warp_matrix, np.array([[0.0, 0.0, 1.0]])), axis=0)
        T = np.array([[2 / self.Image_size, 0, -1],
                      [0, 2 / self.Image_size, -1],
                      [0, 0, 1]])
        theta = np.linalg.inv(T @ warp_matrix @ np.linalg.inv(T))
        theta_inv = np.linalg.inv(theta)
        return theta[:2, :], theta_inv[:2, :]

    def __len__(self):
        return len(self.Dataset['index'])

    def __getitem__(self, idx):
        curr_row = self.Dataset.iloc[idx]
        folder = curr_row['class']
        imgname = curr_row['filename']
        label = curr_row['label']
        super_label = self.super_category[str(int(label))]

        if self.is_train:
            path = os.path.join(self.root, 'train_train', folder, imgname)
        else:
            path = os.path.join(self.root, 'train_test', folder, imgname)

        Img = pil_loader(path, 'RGB')
        Img = np.array(Img)

        Mask = self.getmasks(idx)
        if Mask == None:
            Mask = torch.zeros(size=(40, Img.shape[-2], Img.shape[-1]))

        Img_shape = Img.shape
        if len(Img_shape) == 2:
            Img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        BBox = np.array([0.0, 0.0, self.Image_size - 1.0, self.Image_size - 1.0])
        Img = cv2.resize(Img, (self.Image_size, self.Image_size), interpolation=cv2.INTER_LINEAR)
        Mask = Mask.unsqueeze(0)
        mask_gt_background = torch.full(size=(1, 1, Mask.shape[-2], Mask.shape[-1]),
                                        fill_value=0.1)
        Mask = torch.cat((Mask, mask_gt_background), dim=1)
        Mask = F.interpolate(Mask, size=(self.Image_size, self.Image_size), mode='nearest')
        Mask = Mask.squeeze(0)

        Mask = Mask.permute(1, 2, 0).numpy()
        Mask = np.argmax(Mask, axis=2, keepdims=True).astype(np.uint8)
        Mask = cv2.resize(Mask, (self.Image_size, self.Image_size), interpolation=cv2.INTER_NEAREST)

        if self.is_train == True:
            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    Img, BBox = self.Image_Flip(Img, BBox)

            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle_1 = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            angle_2 = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale_1 = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)
            Scale_2 = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)

            Translation_X_1 = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_X_2 = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_Y_1 = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)
            Translation_Y_2 = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)

            trans_1 = utils.get_transforms(BBox, Scale_1, angle_1, self.Image_size, shift_factor=[Translation_X_1, Translation_Y_1])
            trans_2 = utils.get_transforms(BBox, Scale_2, angle_2, self.Image_size, shift_factor=[Translation_X_2, Translation_Y_2])

            input_1 = cv2.warpAffine(Img, trans_1, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
            input_2 = cv2.warpAffine(Img, trans_2, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            if self.Transform is not None:
                input_1 = self.Transform(input_1)
                input_2 = self.Transform(input_2)

            meta = {
                    'Img': input_1,
                    'Img_pair': input_2,
                    'super_label': super_label,
                    'label': label,
                    'Img_path': path,
                    'BBox': BBox,
                    'trans_1': trans_1,
                    'trans_2': trans_2,
                    'Scale': [Scale_1, Scale_2],
                    'angle': [angle_1, angle_2],
                    'Translation': [Translation_X_1, Translation_Y_1, Translation_X_2, Translation_Y_1]}

            return meta

        else:
            if self.Transform is not None:
                input = self.Transform(Img)

            meta = {
                'Img': input,
                'super_label': super_label,
                'label': label,
                'Mask': Mask,
                'Img_path': path,
                'BBox': BBox,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
            }

            return meta

if __name__ == '__main__':
    import torch
    import argparse

    from Config import cfg

    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--modelDir', help='model directory', type=str, default='./output')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    dataset = PartImage_Dataset(cfg, '../../Data/PartImageNet_Processed', False,  transforms.Compose([transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    for i, meta in enumerate(train_loader):
        img = meta['Img'].numpy().transpose(0, 2, 3, 1)
        Mask = meta['Mask']
        label = meta['label']
        print(meta['super_label'])
        print(label)
        Mask = Mask.numpy()
        Mask = Mask * 6

        img = ((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0

        for i in range(16):
            temp_img = img[i].copy().astype(np.uint8)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            temp_mask = Mask[i].copy().astype(np.uint8)

            cv2.imshow('test1', temp_img)
            cv2.imshow('test2', temp_mask)

            cv2.waitKey(0)
