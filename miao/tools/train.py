'''
1. 读取参数设置
2. 训练模型，保存损失
    - 写入losses_df
    - 将[yml配置文件, 整体val_IC]添加到表格里
'''
from pathlib import Path
import argparse
from train_utils.config import cfg_from_yaml_file, cfg
from train_utils.optimization import build_optimizer, masked_mse, corr_value
from train_utils.train_utils import train_model
from eval_utils.eval_utils import eval_model
from datasets import build_dataloader
import torch
from models import build_network

import models

models.MarketMixer

from torch.optim import Adam

from tensorboardX import SummaryWriter

def parse_config():
    '''
        parser: 参数设置，优先级高于配置文件
        cfg: 配置文件
    '''
    parser = argparse.ArgumentParser(description='arg parser')
    # cfgs/SimpleFeatsMixer/h128_bestmodel.yaml
    # cfgs/MLP_miner/demo.yaml
    parser.add_argument('--cfg_file', type=str, default='cfgs/SimpleFeatsMixer/h128_bestmodel.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--extra_tag', type=str, default='best', required=False, help='exp extra tag')
    parser.add_argument('--save_pred', type=bool, default=False, required=False, help='save predictions')
    parser.add_argument('--early_stop', type=bool, default=False, required=False, help='early stop, defalut epoch > 5')
    parser.add_argument('--save_model', type=bool, default=False, required=False, help='save model')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)

    # 例如: cfg_file = 'cfgs/GRU_miner/exp1.yaml' ---> TAG = 'exp1' && EXP_GROUP_PATH = 'GRU_miner'
    cfg.TAG = Path(args.cfg_file).stem  # 配置文件的文件名
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # 获取实验分组 remove 'cfgs' and 'xxxx.yaml'

    return args, cfg


def main():
    # 读取参数设置
    args, cfg = parse_config()

    args.batch_size = cfg.OPTIMIZATION.NUM_EPOCHS if args.batch_size is None else args.batch_size
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    args.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    args.early_stop = cfg.OPTIMIZATION.EARLY_STOP if args.early_stop is None else args.early_stop
    args.save_model = cfg.OPTIMIZATION.SAVE_MODEL if args.save_model is None else args.save_model

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    report_dir = cfg.ROOT_DIR / 'output' / 'reports' / cfg.EXP_GROUP_PATH
    pred_dir = output_dir / 'preds'

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    report_name = report_dir / f'{cfg.TAG}_{args.extra_tag}.html'

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    print("正在读取数据: ", cfg.DATA_CONFIG.DATA_PATH + ">>>" + cfg.DATA_CONFIG.DATASET)
    train_set, train_loader, val_loader, test_loader = build_dataloader(cfg, args.device)  # 构建数据集
    extra_data = train_set.dataset.extra_data
    dt = extra_data['dt']
    symbols = extra_data['symbols']

    model = build_network(model_cfg=cfg.MODEL, dataset=train_set)  # 构建模型
    if torch.cuda.is_available():
        model.cuda()
    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    loss_fn, val_score_fn = masked_mse, corr_value

    train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        loss_criterion=loss_fn, 
        validate_score=val_score_fn, 
        tb_log=tb_log, 
        num_epochs=args.epochs,
        early_stop=args.early_stop,
        save_model=args.save_model,
    )

    y_pred_pldf = eval_model(cfg, model, test_loader, dt, symbols, report_name)
    if args.save_pred:
        y_pred_pldf.collect().write_parquet(pred_dir / 'y_preds.parquet')

if __name__ == '__main__':
    # 修改运行路径为当前文件路径
    import os
    seed = 1895
    torch.random.manual_seed(seed)
    os.chdir(os.path.dirname(__file__))
    main()