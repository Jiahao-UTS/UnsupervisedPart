import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def segment_to_landmark(score_map, x_map, y_map):
    SumWeight = torch.sum(score_map, dim=(2, 3), keepdim=True)

    X_sum = torch.sum(score_map * x_map, dim=(2, 3), keepdim=True)
    Y_sum = torch.sum(score_map * y_map, dim=(2, 3), keepdim=True)

    X_coord = X_sum / SumWeight
    Y_coord = Y_sum / SumWeight

    coord = torch.cat([X_coord, Y_coord], dim=3)
    coord = coord.squeeze(2)
    return coord

def calculate_NME(MAFL_train_pred, MAFL_train_GT, MAFL_test_pred, MAFL_test_GT):
    scaler_pred = StandardScaler()
    scaler_gt = StandardScaler()

    scaler_pred.fit(MAFL_train_pred.reshape(MAFL_train_pred.shape[0], -1))
    scaler_gt.fit(MAFL_train_GT.reshape(MAFL_train_GT.shape[0], -1))

    MAFL_train_pred = scaler_pred.transform(MAFL_train_pred.reshape(MAFL_train_pred.shape[0], -1))
    MAFL_train_GT = scaler_gt.transform(MAFL_train_GT.reshape(MAFL_train_GT.shape[0], -1))

    MAFL_train_GT = torch.from_numpy(MAFL_train_GT).float()
    MAFL_train_pred = torch.from_numpy(MAFL_train_pred)

    MAFL_test_pred = scaler_pred.transform(MAFL_test_pred.reshape(MAFL_test_pred.shape[0], -1))

    MAFL_test_GT = torch.from_numpy(MAFL_test_GT).float()
    MAFL_test_pred = torch.from_numpy(MAFL_test_pred)

    MAFL_train_pred = MAFL_train_pred.reshape(MAFL_train_pred.shape[0], -1)
    MAFL_train_GT = MAFL_train_GT.reshape(MAFL_train_GT.shape[0], -1)
    MAFL_test_pred = MAFL_test_pred.reshape(MAFL_test_pred.shape[0], -1)
    MAFL_test_GT = MAFL_test_GT.reshape(MAFL_test_GT.shape[0], -1)
    try:
        beta = (MAFL_train_pred.T @ MAFL_train_pred).inverse() @ MAFL_train_pred.T @ MAFL_train_GT
    except:
        beta = (MAFL_train_pred.T @ MAFL_train_pred + torch.eye(MAFL_train_pred.shape[-1],
                                                                dtype=torch.float32)).inverse() @ MAFL_train_pred.T @ MAFL_train_GT
    pred_y = MAFL_test_pred @ beta

    pred_y = pred_y.numpy()
    pred_y = scaler_gt.inverse_transform(pred_y)
    pred_y = torch.from_numpy(pred_y)

    unnormalized_loss = (pred_y - MAFL_test_GT).reshape(MAFL_test_GT.shape[0], 5, 2).norm(dim=-1)
    eye_distance = (
                MAFL_test_GT.reshape(MAFL_test_GT.shape[0], 5, 2)[:, 0, :] - MAFL_test_GT.reshape(MAFL_test_GT.shape[0],
                                                                                                  5, 2)[:, 1, :]).norm(
        dim=-1)
    normalized_loss = (unnormalized_loss / eye_distance.unsqueeze(1)).mean()

    return normalized_loss.item()