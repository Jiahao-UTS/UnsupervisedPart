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


class CUB_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.CUB.FRACTION
        self.Translation_Factor = cfg.CUB.TRANSLATION
        self.Rotation_Factor = cfg.CUB.ROTATION
        self.Scale_Factor = cfg.CUB.SCALE
        self.Flip = cfg.CUB.FLIP

        self.Transform = transform

        if is_train:
            self.Image_path = os.path.join(root, 'train_list.txt')
        else:
            self.Image_path = os.path.join(root, 'test_list.txt')
        self.box_path = os.path.join(root, 'bounding_boxes.txt')
        self.landmark_path = os.path.join(root, "parts", "part_locs.txt")
        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []

        with open(self.Image_path) as f:
            img_list = f.read().splitlines()
            f.close()

        landmark_dict = {}

        box_dict = {}

        with open(self.landmark_path) as f:
            landmark_list = f.read().splitlines()
            f.close()

        for temp_landmark in landmark_list:
            temp_landmark = temp_landmark.split(' ')
            if temp_landmark[0] not in landmark_dict.keys():
                landmark_dict[temp_landmark[0]] = np.zeros((15, 3), dtype=np.float)
            landmark_dict[temp_landmark[0]][int(temp_landmark[1])-1, :] = \
                np.array([float(temp_landmark[2]), float(temp_landmark[3]), float(temp_landmark[4])], dtype=np.float)

        with open(self.box_path) as f:
            box_list = f.read().splitlines()
            f.close()

        for temp_box in box_list:
            temp_box = temp_box.split(' ')

            box_dict[temp_box[0]] = np.array([float(temp_box[1]), float(temp_box[2]),
                                              float(temp_box[3]), float(temp_box[4])], dtype=np.float)

        for temp_path in img_list:
            temp_path = temp_path.split(' ')
            temp_image_path = os.path.join(self.root, 'images', temp_path[1])
            temp_image_number = temp_path[0]
            Data_base.append({'Img': temp_image_path,
                              'box': box_dict[temp_image_number],
                              'landmark': landmark_dict[temp_image_number].copy(),
                              'number': int(temp_image_number)})

        return Data_base

    def Image_Flip(self, Img, bbox):
        Img = cv2.flip(Img, 1)

        width = Img.shape[1]
        bbox[0] = width - 1 - bbox[2] - bbox[0]

        return Img, bbox

    def Create_Occlusion(self, Img):
        Occlusion_width = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_high = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_x = np.random.randint(0, self.Image_size - Occlusion_width)
        Occlusion_y = np.random.randint(0, self.Image_size - Occlusion_high)

        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 0] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 1] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 2] = \
            np.random.randint(0, 256)

        return Img

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
        number = db_slic['number']
        landmark = db_slic['landmark']
        box = db_slic['box']

        Img = cv2.imread(Img_path)
        Img_shape = Img.shape

        # BBox = np.array([0.0, 0.0, Img_shape[1] - 1.0, Img_shape[0] - 1.0])

        Img = cv2.resize(Img, (self.Image_size, self.Image_size))
        BBox = np.array([0.0, 0.0, self.Image_size - 1, self.Image_size - 1])
        landmark[:,:2] = (landmark[:,:2] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)
        box[:2] = (box[:2] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)
        box[2:4] = (box[2:4] * self.Image_size) / np.array([[Img_shape[1], Img_shape[0]]], dtype=np.float)


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

            input_1 = cv2.warpAffine(Img, trans_1, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
            input_2 = cv2.warpAffine(Img, trans_2, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            if self.Transform is not None:
                input_1 = self.Transform(input_1)
                input_2 = self.Transform(input_2)

            meta = {
                    'Img': input_1,
                    'Img_pair': input_2,
                    'box': box,
                    'landmark': landmark,
                    'number': number,
                    'Img_path': Img_path,
                    'BBox': BBox,
                    'trans_1': trans_1,
                    'trans_2': trans_2,
                    'Scale': [Scale_1, Scale_2],
                    'angle': [angle_1, angle_2],
                    'Translation': [Translation_X_1, Translation_Y_1, Translation_X_2, Translation_Y_1]}

            return meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0 * np.pi / 180.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            if self.Transform is not None:
                input = self.Transform(input)

            for i in range(len(landmark)):
                landmark[i, 0:2] = utils.affine_transform(landmark[i, 0:2], trans)
            for i in range(len(landmark)):
                if landmark[i, 0] <= 0 or landmark[i, 0] > self.Image_size:
                    landmark[i, 0] = 0.5
                    landmark[i, 2] = 0
                if landmark[i, 1] <= 0 or landmark[i, 1] > self.Image_size:
                    landmark[i, 1] = 0.5
                    landmark[i, 2] = 0

            meta = {
                'Img': input,
                'Img_path': Img_path,
                'landmark': landmark,
                'number': number,
                'box': box,
                'BBox': BBox,
                'trans': trans,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
            }

            return meta


# if __name__ == '__main__':
#     with open("../../Data/CelebA/list_landmarks_celeba.txt") as f:
#         file_info = f.read().splitlines()[2:]
#         f.close()
#
#     test_info = file_info[77].split()
#     print(test_info)
#     img = cv2.imread("../../Data/CelebA/img_celeba/" + test_info[0])
#     point = []
#     for i in range(10):
#         point.append(test_info[i+1])
#     point = np.array(point, dtype=np.int).reshape(5,2)
#     for i in point:
#         cv2.circle(img, (i[0], i[1]), 4, (0, 0, 255), -1)
#         cv2.imshow("test", img)
#
#     mean_face = np.load("../Config/init_98.npz")["init_face"].reshape(98, 2)
#     mean_face_five = np.stack([mean_face[96], mean_face[97], mean_face[54], mean_face[76], mean_face[82]])
#
#     affine_matrix = transformation_from_points(mean_face_five, point)
#     perspective_matrix = get_perspective_matrix(mean_face_five, point)
#
#     print(perspective_matrix)
#
#     warped_face = mean_face.copy()
#     warped_face = np.array(affine_transform(warped_face, perspective_matrix, 98))
#     for i in warped_face:
#         cv2.circle(img, (int(i[0]), int(i[1])), 4, (0, 255, 0), -1)
#
#     cv2.imshow("test", img)
#     cv2.waitKey(0)

if __name__ == '__main__':
    import torch
    import argparse

    from Config import cfg
    from Config import update_config

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

    dataset = CUB_Dataset(cfg, '../../../unsupervised landmark/Data/CUB', False,  transforms.Compose([transforms.ToTensor(),
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
        # img_pair = meta['Img_pair'].numpy().transpose(0, 2, 3, 1)

        img = ((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0
        # img_pair = ((img_pair * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255.0

        for i in range(16):
            temp_img = img[i].copy().astype(np.uint8)
            # temp_img_pair = img_pair[i].copy().astype(np.uint8)

            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            # temp_img_pair = cv2.cvtColor(temp_img_pair, cv2.COLOR_BGR2RGB)

            cv2.imshow('test1', temp_img)
            # cv2.imshow('test2', temp_img_pair)

            cv2.waitKey(0)