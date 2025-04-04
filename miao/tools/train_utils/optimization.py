import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import numpy as np

def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    else:
        raise NotImplementedError
    return optimizer


mse = torch.nn.MSELoss() 
def masked_mse(y_pred, y_true, mask):
    y_pred = y_pred * mask
    y_true = y_true * mask
    mask_ratio = mask.sum() / mask.numel()
    loss = mse(y_pred, y_true) / mask_ratio
    return loss

def corr_value(outputs, labels, masks=None):
    '''自定义模型选择的metrics'''
    if masks is not None:
        outputs, labels = [torch.masked_select(x, masks.bool()) for x in [outputs, labels]]
    outputs = outputs.flatten().detach().numpy()
    labels = labels.flatten().detach().numpy()
    if labels.sum() != 0 and outputs.sum() != 0:
        corr_value = np.corrcoef(outputs, labels)
        return corr_value[0,1]
    else:
        return 0.
    
if __name__ == '__main__':
    test_arr_1 = np.array([0,0,0])
    test_arr_2 = np.array([1,2,3])
    corr_value = np.corrcoef(test_arr_1, test_arr_2)
    print(corr_value[0,1])
    print(np.isnan(corr_value[0,1]))