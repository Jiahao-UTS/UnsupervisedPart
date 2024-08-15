import time
import logging
import os

import numpy as np
import torch

from torch.nn import functional as F
from utils import AverageMeter

logger = logging.getLogger(__name__)


def train(config, train_loader, model, vgg, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    A_loss_average = AverageMeter()
    H_loss_average = AverageMeter()
    C_loss_average = AverageMeter()
    S_loss_average = AverageMeter()
    loss_average = AverageMeter()

    model.train()
    model.module.backbone1.eval()
    vgg.eval()

    end = time.time()

    for i, meta in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = meta['Img'].cuda()
        input_pair = meta['Img_pair'].cuda()

        if config.DATASET.DATASET == 'PartImage':
            label = meta['label'].cuda().long()
            Area_loss, Highlevel_loss, Concentration_loss, Semantic_loss = model(input, label, input_pair, vgg, True)
        else:
            Area_loss, Highlevel_loss, Concentration_loss, Semantic_loss = model(input, input_pair, vgg, True)

        loss = 1.0 * Highlevel_loss + 0.3 * Concentration_loss + 0.5 * Area_loss + 0.01 * Semantic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_average.update(loss.item(), input.size(0))
        H_loss_average.update(Highlevel_loss.item(), input.size(0))
        C_loss_average.update(Concentration_loss.item(), input.size(0))
        S_loss_average.update(Semantic_loss.item(), input.size(0))
        A_loss_average.update(Area_loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'H_loss: {H_loss.val:.5f} ({H_loss.avg:.5f})\t' \
                  'C_loss: {C_loss.val:.5f} ({C_loss.avg:.5f})\t' \
                  'S_loss: {S_loss.val:.5f} ({S_loss.avg:.5f})\t' \
                  'A_loss: {A_loss.val:.5f} ({A_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, C_loss=C_loss_average,
                S_loss=S_loss_average, loss=loss_average,
                H_loss=H_loss_average, A_loss=A_loss_average)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_average.val, global_steps)
            writer.add_scalar('C_loss', C_loss_average.val, global_steps)
            writer.add_scalar('H_loss', H_loss_average.val, global_steps)
            writer.add_scalar('S_loss', S_loss_average.val, global_steps)
            writer.add_scalar('A_loss', A_loss_average.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

