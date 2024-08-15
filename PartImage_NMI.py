import argparse
import math

from Config import cfg
from Config import update_config

from utils import create_logger
from PartImage_Model import UnsupervisedPart_PartImage
from Dataloader import PartImage_Dataset
import torch.nn.functional as F

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix

import torch
import numpy as np
import pprint

import torchvision.transforms as transforms

color = np.array([[255, 0, 0],
                  [0, 255, 0],
                  [0, 0, 255],
                  [255, 0, 255],
                  [0, 0, 0]])


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='./Model/PartImage.pth')
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def adjusted_rand_score_overflow(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    (tn, fp), (fn, tp) = (tn / 1e8, fp / 1e8), (fn / 1e8, tp / 1e8)
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

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

    # 声明模型
    model = UnsupervisedPart_PartImage(cfg.MODEL.NUM_Part, cfg.MODEL.OUT_DIM, cfg.TRAIN.PRE, cfg)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = PartImage_Dataset(
        cfg, cfg.PartImage.ROOT, False,
        transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
    )

    # 验证数据迭代器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = args.checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.module.load_state_dict(checkpoint)

    all_nmi_preds = []
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    with torch.no_grad():
        for i, meta in enumerate(valid_loader):
            input = meta['Img'].cuda()
            mask_gt = meta['Mask'].unsqueeze(1)
            label = meta['label'].long()
            mask_gt = F.interpolate(mask_gt, scale_factor=0.5, mode='nearest')
            bs = input.size(0)

            # 2.----------------------------------------------------------------------
            mask_w_bg = model(input, label)
            # 3.----------------------------------------------------------------------
            mask = mask_w_bg[:, :-1]

            pred_parts_loc = torch.argmax(mask.cpu(), dim=1).view(bs, -1).numpy()
            pred_parts_loc[pred_parts_loc!=4] = pred_parts_loc[pred_parts_loc!=4] + label.float().numpy() * 4.0
            all_nmi_preds.append(pred_parts_loc.reshape(-1))

            pred_parts_loc_w_bg = torch.argmax(mask_w_bg.cpu(), dim=1).view(bs, -1).numpy()
            pred_parts_loc_w_bg[pred_parts_loc_w_bg!=4] = pred_parts_loc_w_bg[pred_parts_loc_w_bg!=4] + label.float().numpy() * 4.0
            mask_gt = mask_gt.view(bs, -1)
            mask_gt = mask_gt.cpu()
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.reshape(-1))

            all_nmi_gts.append(mask_gt.view(-1).numpy())

        all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)
        all_nmi_preds_w_bg = np.concatenate(all_nmi_preds_w_bg, axis=0)

        nmi1 = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds_w_bg) * 100
        ari1 = adjusted_rand_score_overflow(all_nmi_gts, all_nmi_preds_w_bg) * 100

        print(nmi1, ari1)

if __name__ == '__main__':
    main_function()

