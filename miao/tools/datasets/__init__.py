from .data_utils import MultiFeatsDataset, CSDataset
from torch.utils.data import DataLoader

__all__ = {
    'stock_mix_data': MultiFeatsDataset,
    'cs_dataset': CSDataset,
    'cs_dataset_la240': CSDataset,  # 仅用于检查模型结构有无错误，特征是 la240（未来数据）
}

def build_dataloader(cfg, device):
    dataset_name = cfg.DATA_CONFIG.DATASET   # 获取数据集名称
    data_path = cfg.DATA_CONFIG.DATA_PATH + dataset_name
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE
    dataset = __all__[dataset_name]()  # 根据数据集名称获取数据集类
    dataset.loaddata(data_path, device)  # 加载数据集

    train_set, val_set, test_set = dataset.get_splits()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_set, train_loader, val_loader, test_loader

