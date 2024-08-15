import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backbone import Get_vit, conv_ln, ArcMarginProduct, load_DINO

class UnsupervisedPart_CUB(nn.Module):
    def __init__(self, num_point, d_model, pretrain,
                 cfg):
        super(UnsupervisedPart_CUB, self).__init__()

        self.num_point = num_point
        self.d_model = d_model
        self.background = cfg.MODEL.BACKGROUND

        self.Reconstruct_Block = nn.Sequential(
            conv_ln(64, 64, 1),
            conv_ln(64, 64, 1),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            conv_ln(64, 64, 1),
            conv_ln(64, 64, 1),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            conv_ln(64, 64, 1),
            nn.Conv2d(64, 3, (1, 1), (1, 1))
        )

        self.compress_layer = self.compress_layer = nn.Sequential(
            nn.Conv2d(384, 256, (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (1, 1), (1, 1)),
        )
        
        self.ArcMarginProduct = ArcMarginProduct(d_model, self.num_point-self.background, self.num_point-self.background)

        self.apply(self._init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        x_map = torch.arange(cfg.MODEL.IMG_SIZE // 8, dtype=torch.float32).cuda() / (cfg.MODEL.IMG_SIZE // 8) * 2.0 - 1.0
        self.x_map = x_map.unsqueeze(0).repeat(cfg.MODEL.IMG_SIZE // 8, 1)
        self.x_map = self.x_map.unsqueeze(0).unsqueeze(0)

        y_map = torch.arange(cfg.MODEL.IMG_SIZE // 8, dtype=torch.float32).cuda() / (cfg.MODEL.IMG_SIZE // 8) * 2.0 - 1.0
        self.y_map = y_map.unsqueeze(1).repeat(1, cfg.MODEL.IMG_SIZE // 8)
        self.y_map = self.y_map.unsqueeze(0).unsqueeze(0)

        self.backbone1 = load_DINO("small", pretrain, 8)
        self.Transformer = Get_vit(num_point, 128, 16, 256, 6, 8, 64)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def Concentration_loss(self, pred, epsilon=0.5):

        SumWeight = torch.sum(pred, dim=(2, 3), keepdim=True) + 1e-8

        X_sum = torch.sum(pred * self.x_map, dim=(2, 3), keepdim=True)
        Y_sum = torch.sum(pred * self.y_map, dim=(2, 3), keepdim=True)

        # [B, N, 1, 1]
        X_coord = X_sum / SumWeight
        Y_coord = Y_sum / SumWeight

        X_var = torch.square(self.x_map - X_coord)
        Y_var = torch.square(self.y_map - Y_coord)

        X_var = torch.sum(X_var * pred / SumWeight, dim=(1, 2, 3))
        Y_var = torch.sum(Y_var * pred / SumWeight, dim=(1, 2, 3))

        size_constrain = 1 / (1 + SumWeight / epsilon)
        size_constrain = torch.sum(size_constrain, dim=(1, 2, 3))

        return torch.mean(X_var + Y_var), torch.mean(size_constrain)

    def high_level_loss(self, input_feature, img, vgg, return_img=False):
        Bs = img.size(0)

        R_img = self.Reconstruct_Block(input_feature)
        vgg_features = vgg(torch.cat((R_img, img), dim=0))

        loss_weight = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0]
        loss = 0

        for i in range(len(vgg_features)):
            feature_diff = (vgg_features[i][0:Bs] - vgg_features[i][Bs:])
            value = torch.abs(feature_diff).mean()
            loss += value * loss_weight[i]

        if return_img is False:
            return loss
        else:
            return loss, R_img

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input, input_pair=None, VGG=None, trainable=False):
        # 产生特征图
        Large_input = input
        Large_input_pair = input_pair
        input = F.interpolate(input, scale_factor=0.5)

        Local_Features = self.backbone1(Large_input)
        Local_Features = self.compress_layer(Local_Features)
        Global_Feature = self.Transformer(input)

        if trainable:
            input_pair = F.interpolate(input_pair, scale_factor=0.5)

            Local_Features_pair = self.backbone1(Large_input_pair)
            Local_Features_pair = self.compress_layer(Local_Features_pair)
            Global_Feature_pair = self.Transformer(input_pair)

            output_distribution_raw_pair = torch.einsum("bqc,bchw->bqhw", Global_Feature, Local_Features_pair)
            output_distribution_raw_pair = F.softmax(output_distribution_raw_pair * 0.8, dim=1)
            R_img_pair = torch.einsum("bln,bnc->blc", Global_Feature.permute(0, 2, 1), output_distribution_raw_pair.flatten(2))

            output_distribution_pair_raw = torch.einsum("bqc,bchw->bqhw", Global_Feature_pair, Local_Features)
            output_distribution_pair_raw = F.softmax(output_distribution_pair_raw * 0.8, dim=1)
            R_img = torch.einsum("bln,bnc->blc", Global_Feature_pair.permute(0, 2, 1), output_distribution_pair_raw.flatten(2))

            Bs = R_img_pair.size(0)
            R_img_pair = R_img_pair.view(Bs, self.d_model, 32, 32)
            R_img = R_img.view(Bs, self.d_model, 32, 32)

            Highlevel_loss_raw_pair = self.high_level_loss(R_img_pair, input_pair, VGG)
            concentration_loss_raw_pair, area_loss_raw_pair = self.Concentration_loss(output_distribution_raw_pair[:, :self.num_point-self.background, :, :])
            Highlevel_loss_pair_raw = self.high_level_loss(R_img, input, VGG)
            concentration_loss_pair_raw, area_loss_pair_raw = self.Concentration_loss(output_distribution_pair_raw[:, :self.num_point-self.background, :, :])

            Highlevel_loss = (Highlevel_loss_raw_pair + Highlevel_loss_pair_raw) / 2.0
            Concentration_loss = (concentration_loss_raw_pair + concentration_loss_pair_raw) / 2.0
            area_loss = (area_loss_raw_pair + area_loss_pair_raw) / 2.0
            semantic_loss = self.ArcMarginProduct(Global_Feature[:, :self.num_point-self.background, :]) \
                            + self.ArcMarginProduct(Global_Feature_pair[:, :self.num_point-self.background, :])
            semantic_loss = semantic_loss / 2.0

            return area_loss, Highlevel_loss, Concentration_loss, semantic_loss

        else:
            Local_Features = F.interpolate(Local_Features, scale_factor=4.0, mode='bilinear')
            output_distribution_raw_raw = torch.einsum("bqc,bchw->bqhw", Global_Feature, Local_Features)
            output_distribution_raw_raw = F.softmax(output_distribution_raw_raw * 0.8, dim=1)
            return output_distribution_raw_raw
