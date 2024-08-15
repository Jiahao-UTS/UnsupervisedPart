import argparse
import math

from Config import cfg
from Config import update_config

from utils import create_logger
from CUB_Model import UnsupervisedPart_CUB
from Dataloader import CUB_Dataset
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


import torch
import numpy as np
import pprint

import torchvision.transforms as transforms

color = np.array([[255, 0, 0],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 0, 255],
                  [0, 255, 255],
                  [255, 255, 0],
                  [128, 128, 0],
                  [128, 0, 128]])


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='./Model/CUB.pth')
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args

def main_function():
    # 获得参数
    args = parse_args()
    # 更新参数
    update_config(cfg, args)
    # 创建日志文件目录
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    # 输入输入参数
    logger.info(pprint.pformat(args))
    # 输入CFG配置参数
    logger.info(cfg)

    # 配置CUDNN参数
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = UnsupervisedPart_CUB(cfg.MODEL.NUM_Part, cfg.MODEL.OUT_DIM, cfg.TRAIN.PRE, cfg)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = CUB_Dataset(
        cfg, cfg.CUB.ROOT,  False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    # 验证数据迭代器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = args.checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.module.load_state_dict(checkpoint)

    # 切换模型为预测状态
    model.eval()

    all_nmi_preds = []
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    # torch no_grad()中的运算均不会计算梯度
    with torch.no_grad():
        for i, meta in enumerate(valid_loader):
            # 计算输出
            input = meta['Img'].cuda()
            landmark = meta['landmark']

            mask_w_bg = model(input)
            mask = mask_w_bg[:, :4]
            visible = landmark[:, :, 2] > 0.5

            points = landmark[:, :, 0:2].unsqueeze(2).clone()

            points[:, :, :, 0] /= input.shape[-1]  # W
            points[:, :, :, 1] /= input.shape[-2]  # H
            assert points.min() > -1e-7 and points.max() < 1 + 1e-7
            points = points * 2 - 1

            pred_parts_loc = F.grid_sample(mask.float(), points.float().cuda(), mode='nearest', align_corners=False)
            pred_parts_loc = torch.argmax(pred_parts_loc, dim=1).squeeze(2)
            pred_parts_loc = pred_parts_loc[visible]
            all_nmi_preds.append(pred_parts_loc.cpu().numpy())

            pred_parts_loc_w_bg = F.grid_sample(mask_w_bg.float(), points.float().cuda(), mode='nearest', align_corners=False)
            pred_parts_loc_w_bg = torch.argmax(pred_parts_loc_w_bg, dim=1).squeeze(2)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg[visible]
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())

            gt_parts_loc = torch.arange(points.shape[1]).unsqueeze(0).repeat(points.shape[0], 1)
            gt_parts_loc = gt_parts_loc[visible]
            all_nmi_gts.append(gt_parts_loc.cpu().numpy())

        all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)
        all_nmi_preds = np.concatenate(all_nmi_preds, axis=0)
        all_nmi_preds_w_bg = np.concatenate(all_nmi_preds_w_bg, axis=0)

        nmi1 = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds_w_bg) * 100
        ari1 = adjusted_rand_score(all_nmi_gts, all_nmi_preds_w_bg) * 100

        nmi2 = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds) * 100
        ari2 = adjusted_rand_score(all_nmi_gts, all_nmi_preds) * 100

        print(nmi1, ari1, nmi2, ari2)




if __name__ == '__main__':
    main_function()

