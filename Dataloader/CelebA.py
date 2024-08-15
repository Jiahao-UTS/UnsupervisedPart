import copy

import cv2
import numpy as np
import os
import utils

from torch.utils.data import Dataset


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


class Cele_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, test_flag=None, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.CELE.FRACTION
        self.Translation_Factor = cfg.CELE.TRANSLATION
        self.Rotation_Factor = cfg.CELE.ROTATION
        self.Scale_Factor = cfg.CELE.SCALE
        self.Flip = cfg.CELE.FLIP

        self.Transform = transform

        if is_train:
            self.annotation_file = os.path.join(root, 'cele_train_lm.txt')
        else:
            if test_flag == 'val':
                self.annotation_file = os.path.join(root, 'MAFL_train_lm.txt')
            elif test_flag == 'test':
                self.annotation_file = os.path.join(root, 'MAFL_test_lm.txt')
            else:
                raise NotImplementedError

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_info = temp_info.split(',')

            temp_name = os.path.join(self.root, 'img_celeba', temp_info[0])
            points = np.array([float(temp_info[1]), float(temp_info[2]), float(temp_info[3]), float(temp_info[4]),
                               float(temp_info[5]), float(temp_info[6]), float(temp_info[7]), float(temp_info[8]),
                               float(temp_info[9]), float(temp_info[10])])
            points = points.reshape((5, 2))

            Data_base.append({'Img': temp_name,
                              'points': points})

        return Data_base

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
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        points = db_slic['points']

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape

        Img = cv2.resize(Img, (self.Image_size, self.Image_size))
        points[:, 0] = points[:, 0] / Img_shape[1] * self.Image_size
        points[:, 1] = points[:, 1] / Img_shape[0] * self.Image_size
        BBox = np.array([0.0, 0.0, self.Image_size, self.Image_size])

        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

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

            for i in range(len(points)):
                points[i, 0:2] = utils.affine_transform(points[i, 0:2], trans_1)

            input_1 = cv2.warpAffine(Img, trans_1, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
            input_2 = cv2.warpAffine(Img, trans_2, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            if self.Transform is not None:
                input_1 = self.Transform(input_1)
                input_2 = self.Transform(input_2)

            meta = {
                    'Img': input_1,
                    'Img_pair': input_2,
                    'points': points,
                    'Img_path': Img_path,
                    'BBox': BBox,
                    'trans_1': trans_1,
                    'trans_2': trans_2,
                    'Scale': [Scale_1, Scale_2],
                    'angle': [angle_1, angle_2],
                    'Translation': [Translation_X_1, Translation_Y_1, Translation_X_2, Translation_Y_1]}

            return meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            for i in range(len(points)):
                points[i, 0:2] = utils.affine_transform(points[i, 0:2], trans)

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            if self.Transform is not None:
                input = self.Transform(input)

            meta = {
                'Img': input,
                'points': points,
                'Img_path': Img_path,
                'BBox': BBox,
                'trans': trans,
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

    dataset = Cele_Dataset(cfg, '../../Data/CelebA', True,  True, transforms.Compose([transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    for i, meta in enumerate(train_loader):
        img = meta['Img'].numpy().transpose(0, 2, 3, 1)
        img_pair = meta['Img_pair'].numpy().transpose(0, 2, 3, 1)
        points = meta['points'].numpy()


        img = ((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0
        img_pair = ((img_pair * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0

        for i in range(16):
            temp_img = img[i].copy().astype(np.uint8)
            point = points[i].copy()
            temp_img_pair = img_pair[i].copy().astype(np.uint8)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            temp_img_pair = cv2.cvtColor(temp_img_pair, cv2.COLOR_BGR2RGB)

            for temp in point:
                cv2.circle(temp_img, (int(temp[0] + 0.5), int(temp[1] + 0.5)), 3, (0, 255, 0), -1)

                cv2.imshow('test1', temp_img)
                cv2.waitKey(0)
            cv2.imshow('test2', temp_img_pair)

            cv2.waitKey(0)
