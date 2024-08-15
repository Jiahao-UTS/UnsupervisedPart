import numpy as np
import time
import torch
import logging
import os
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix

from utils import AverageMeter, segment_to_landmark, calculate_NME

logger = logging.getLogger(__name__)

def adjusted_rand_score_overflow(labels_true, labels_pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    (tn, fp), (fn, tp) = (tn / 1e8, fp / 1e8), (fn / 1e8, tp / 1e8)
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

def validate_CUB(config, val_loader, model, vgg, writer_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    landmark_gt_list = []
    box_list = []

    all_nmi_preds = []
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    model.eval()
    vgg.eval()

    with torch.no_grad():
        end = time.time()
        for i, meta in enumerate(val_loader):
            #1.----------------------------------------------------------------------
            input = meta['Img'].cuda()
            landmark = meta['landmark']
            landmark_gt_list.append(landmark)
            bbox = meta['box'].numpy()[0]
            box_list.append(bbox)

            #2.----------------------------------------------------------------------

            data_time.update(time.time() - end)
            mask_w_bg = model(input)
            batch_time.update(time.time() - end)
            end = time.time()

            #3.----------------------------------------------------------------------
            mask = mask_w_bg[:, :-1]
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

            pred_parts_loc_w_bg = F.grid_sample(mask_w_bg.float(), points.float().cuda(), mode='nearest',
                                                align_corners=False)
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

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('FG_NMI', nmi2, global_steps)
        writer.add_scalar('FG_ARI', ari2, global_steps)
        writer.add_scalar('NMI', nmi1, global_steps)
        writer.add_scalar('ARI', ari1, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        msg = 'FG_NMI: ({FG_NMI:.5f})\t' \
              'FG_ARI: ({FG_ARI:.5f})\t' \
              'NMI: ({NMI:.5f})\t' \
              'ARI: ({ARI:.5f})\t'.format(
              FG_NMI=nmi2, FG_ARI=ari2, NMI=nmi1, ARI=ari1)
        logger.info(msg)
        return nmi1

def validate_PartImage(config, val_loader, model, vgg, writer_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # all_nmi_preds = []
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    model.eval()
    vgg.eval()

    with torch.no_grad():
        end = time.time()

        for i, meta in enumerate(val_loader):

            #1.----------------------------------------------------------------------

            input = meta['Img'].cuda()
            mask_gt = meta['Mask'].unsqueeze(1)
            label = meta['label'].cuda().long()
            mask_gt = F.interpolate(mask_gt, scale_factor=0.5, mode='nearest')
            bs = input.size(0)

            #2.----------------------------------------------------------------------

            data_time.update(time.time() - end)
            mask_w_bg = model(input, label)
            batch_time.update(time.time() - end)
            end = time.time()

            #3.----------------------------------------------------------------------
            pred_parts_loc_w_bg = torch.argmax(mask_w_bg, dim=1).view(bs, -1)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg.cpu().numpy()

            label = label.float().cpu().numpy()

            for idx in range(bs):
                pred_parts_loc_w_bg[idx][pred_parts_loc_w_bg[idx]!= 4] = pred_parts_loc_w_bg[idx][
                                                                    pred_parts_loc_w_bg[idx] != 4] + label[idx] * 4.0
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.reshape(-1))

            mask_gt = mask_gt.view(bs, -1)
            mask_gt = mask_gt.cpu()
            all_nmi_gts.append(mask_gt.view(-1).numpy())

        all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)
        all_nmi_preds_w_bg = np.concatenate(all_nmi_preds_w_bg, axis=0)

        nmi1 = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds_w_bg) * 100
        ari1 = adjusted_rand_score_overflow(all_nmi_gts, all_nmi_preds_w_bg) * 100

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('NMI', nmi1, global_steps)
        writer.add_scalar('ARI', ari1, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        msg = 'NMI: ({NMI:.5f})\t' \
              'ARI: ({ARI:.5f})\t'.format(
               NMI=nmi1, ARI=ari1)
        logger.info(msg)
        return nmi1


def validate_CelebA(config, val_loader, test_loader, model, vgg, writer_dict=None):
    model.eval()
    vgg.eval()

    x_map = torch.arange(128, dtype=torch.float32)
    x_map = x_map.unsqueeze(0).repeat(128, 1)
    x_map = x_map.unsqueeze(0).unsqueeze(0)

    y_map = torch.arange(128, dtype=torch.float32)
    y_map = y_map.unsqueeze(1).repeat(1, 128)
    y_map = y_map.unsqueeze(0).unsqueeze(0)

    val_landmark_list = []
    val_GT_list = []

    test_landmark_list = []
    test_GT_list = []

    with torch.no_grad():
        for i, meta in enumerate(val_loader):
            input = meta['Img'].cuda()
            ground_truth = meta['points'].numpy()

            score_map = model(input)

            landmark = segment_to_landmark(score_map.cpu()[:, :-1, :, :], x_map, y_map).numpy()
            val_landmark_list.append(landmark)
            val_GT_list.append(ground_truth)

        for i, meta in enumerate(test_loader):
            input = meta['Img'].cuda()
            ground_truth = meta['points'].numpy()

            score_map = model(input)

            landmark = segment_to_landmark(score_map.cpu()[:, :-1, :, :], x_map, y_map).numpy()

            test_landmark_list.append(landmark)
            test_GT_list.append(ground_truth)

        val_landmark_list = np.concatenate(val_landmark_list, axis=0)
        val_GT_list = np.concatenate(val_GT_list, axis=0)
        test_landmark_list = np.concatenate(test_landmark_list, axis=0)
        test_GT_list = np.concatenate(test_GT_list, axis=0)

        NME = calculate_NME(val_landmark_list, val_GT_list, test_landmark_list, test_GT_list) * 100.0

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer_dict['valid_global_steps'] = global_steps + 1
        writer.add_scalar('NME', NME, global_steps)

        msg = 'NME: ({NME:.5f})\t'.format(NME=NME)
        logger.info(msg)
        return NME