'''
1. 读取参数设置
2. 训练模型，保存损失
    - 写入losses_df
    - 将[yml配置文件, 整体val_IC]添加到表格里
'''
from pathlib import Path
import argparse
from utils.config import cfg_from_yaml_file, cfg
from utils.train_eval_test import train_model, evaluate_model, test_model, calculate_metrics
from datasets import build_dataloader
import torch
from models import build_network
from torch.optim import Adam
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import pandas as pd

def parse_config():
    '''
        parser: 参数设置，优先级高于配置文件
        cfg: 配置文件
    '''
    parser = argparse.ArgumentParser(description='arg parser')
    
    parser.add_argument('--cfg_file', type=str, default='cfgs/StockNet/ST-GraphAttn.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    # parser.add_argument('--extra_tag', type=str, default='ex_tag', required=False, help='exp extra tag')
    parser.add_argument('--save_pred', type=bool, default=False, required=False, help='save predictions')
    parser.add_argument('--early_stop', type=bool, default=False, required=False, help='early stop, defalut epoch > 5')
    parser.add_argument('--save_model', type=bool, default=False, required=False, help='save model')
    parser.add_argument('--seed', type=int, default=1895, help='random seed')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    # 例如: cfg_file = 'cfgs/GRU_miner/exp1.yaml' ---> TAG = 'exp1' && EXP_GROUP_PATH = 'GRU_miner'
    cfg.TAG = Path(args.cfg_file).stem  # 配置文件的文件名
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # 获取实验分组 remove 'cfgs' and 'xxxx.yaml'

    return args, cfg

def main():
    # 读取参数设置
    args, cfg = parse_config()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE if args.batch_size is None else args.batch_size
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    args.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    args.early_stop = cfg.OPTIMIZATION.EARLY_STOP if args.early_stop is None else args.early_stop
    args.save_model = cfg.OPTIMIZATION.SAVE_MODEL if args.save_model is None else args.save_model

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG 
    ckpt_dir = output_dir / 'ckpt'
    pred_dir = output_dir / 'preds'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    train_data, val_data, test_data = build_dataloader(cfg, args.device)  # 构建数据集


    model = build_network(model_cfg=cfg.MODEL)  # 构建模型
    if torch.cuda.is_available():
        model.cuda()

    print("正在读取数据: ", cfg.DATA_CONFIG.DATA_PATH + ">>>" + cfg.DATA_CONFIG.DATASET + cfg.MODEL.NAME)

    optimizer = Adam(model.parameters(), lr=cfg.OPTIMIZATION.LR)

    # 训练模型
    loss_values, val_loss_values, IC_values = train_model(
        model=model,
        optimizer = optimizer,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        lr=cfg.OPTIMIZATION.LR
    )

    # 记录训练过程中的损失和IC值到tensorboard
    for epoch in range(len(loss_values)):
        tb_log.add_scalar('Loss/train', loss_values[epoch], epoch)
        tb_log.add_scalar('Loss/val', val_loss_values[epoch], epoch)
        tb_log.add_scalar('Metrics/IC', IC_values[epoch], epoch)
    tb_log.close()

    # 测试模型
    y_pred, y_true, mse_loss, mae_loss, rmse_loss, mape_loss = test_model(model, test_data)
    
    # 计算评价指标
    y_pred, y_true = [t.cpu() for t in y_pred], [t.cpu() for t in y_true]
    ic, rank_ic, icir, rankicir = calculate_metrics(y_pred, y_true)
    
    # 将测试指标写入csv文件
    losses_df = pd.DataFrame({
        'MSE Loss': [mse_loss],
        'MAE Loss': [mae_loss],
        'RMSE Loss': [rmse_loss],
        "MAPE Loss": [mape_loss],
        "IC": [ic],
        "RANK IC": [rank_ic],
        "ICIR": [icir],
        "RANK ICIR": [rankicir]
    }).round(8)
    losses_df.to_csv(output_dir / 'test_losses.csv', index=False)
    
    # 保存预测结果
    if args.save_pred:
        pred_df = pd.DataFrame({
            'pred': y_pred.cpu().numpy().flatten(),
            'true': y_true.cpu().numpy().flatten()
        })
        pred_df.to_parquet(pred_dir / 'y_preds.parquet')

    # 保存模型
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_values': loss_values,
            'val_loss_values': val_loss_values,
            'IC_values': IC_values,
            'config': cfg,
            'args': args
        }, ckpt_dir / 'model.pt')

    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(loss_values) + 1),
        'train_loss': loss_values,
        'val_loss': val_loss_values,
        'IC': IC_values
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    return 

if __name__ == '__main__':
    # 修改运行路径为当前文件路径
    import os
    os.chdir(os.path.dirname(__file__))
    main()