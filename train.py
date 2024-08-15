import argparse

from Config import cfg
from Config import update_config

from utils import create_logger, save_checkpoint
from CUB_Model import UnsupervisedPart_CUB
from PartImage_Model import UnsupervisedPart_PartImage
from CelebA_Model import UnsupervisedPart_CelebA
from Dataloader import Cele_Dataset, CUB_Dataset, PartImage_Dataset
from utils import get_optimizer
from tools import train, validate_CUB, validate_PartImage, validate_CelebA
from backbone import Vgg19

from tensorboardX import SummaryWriter

import torch
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignmengt, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.DATASET.DATASET == 'CUB':
        model = UnsupervisedPart_CUB(cfg.MODEL.NUM_Part, cfg.MODEL.OUT_DIM, cfg.TRAIN.PRE, cfg)
    elif cfg.DATASET.DATASET == 'CelebA':
        model = UnsupervisedPart_CelebA(cfg.MODEL.NUM_Part, cfg.MODEL.OUT_DIM, cfg.TRAIN.PRE, cfg)
    elif cfg.DATASET.DATASET == 'PartImage':
        model = UnsupervisedPart_PartImage(cfg.MODEL.NUM_Part, cfg.MODEL.OUT_DIM, cfg.TRAIN.PRE, cfg)
    else:
        raise ValueError('Wrong Dataset')

    vgg = Vgg19()
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    vgg = torch.nn.DataParallel(vgg, device_ids=cfg.GPUS).cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if cfg.DATASET.DATASET == 'CelebA':
        train_dataset = Cele_Dataset(
            cfg, cfg.CELE.ROOT, True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_dataset = Cele_Dataset(
            cfg, cfg.CELE.ROOT, False, 'val',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        test_dataset = Cele_Dataset(
            cfg, cfg.CELE.ROOT, False, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif cfg.DATASET.DATASET == 'CUB':
        train_dataset = CUB_Dataset(
            cfg, cfg.CUB.ROOT,  True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_dataset = CUB_Dataset(
            cfg, cfg.CUB.ROOT,  False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif cfg.DATASET.DATASET == 'PartImage':
        train_dataset = PartImage_Dataset(
            cfg, cfg.PartImage.ROOT, True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_dataset = PartImage_Dataset(
            cfg, cfg.PartImage.ROOT, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise NotImplementedError

    if cfg.DATASET.DATASET == 'CelebA':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

    if (cfg.DATASET.DATASET == 'CUB') or (cfg.DATASET.DATASET == 'PartImage'):
        best_perf = -1.0
    else:
        best_perf = 100
    last_epoch = -1

    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for para in model.module.backbone1.parameters():
        para.requires_grad = False

    for epoch in range(begin_epoch, begin_epoch + cfg.TRAIN.NUM_EPOCH):
        train(cfg, train_loader, model, vgg, optimizer, epoch, writer_dict)
        if cfg.DATASET.DATASET == 'CUB':
            perf_indicator = validate_CUB(
                cfg, valid_loader, model, vgg, writer_dict
            )
        elif cfg.DATASET.DATASET == 'PartImage':
            perf_indicator = validate_PartImage(
                cfg, valid_loader, model, vgg, writer_dict
            )
        elif cfg.DATASET.DATASET == 'CelebA':
            perf_indicator = validate_CelebA(
            cfg, valid_loader, test_loader, model, vgg, writer_dict
            )

        if (cfg.DATASET.DATASET == 'CUB') or (cfg.DATASET.DATASET == 'PartImage'):
            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
        else:
            if perf_indicator <= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        lr_scheduler.step()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main_function()