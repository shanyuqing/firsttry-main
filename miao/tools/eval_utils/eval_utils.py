import torch
from tqdm import trange
import pandas as pd
import polars as pl

import sys
sys.path.append('/home/yuheng/mydata_mgmt/nerv')
from common.html import HtmlBuilder, run_dump_report
from common.pl_util import daily_rel, pl_scatter, pdist, analysis

# 1.预测
# 2.合并预测结果
# 3.通过nerv生成一个html
def eval_model(cfg, model, test_loader, dt, symbols, report_name):
    raw_data_path = cfg.DATA_CONFIG.RAW_DATA_PATH
    model.eval()  # 设置模型为评估模式
    y_pred = []
    full_df = pl.scan_parquet(raw_data_path)
    # 关闭梯度计算以加快推理速度
    with torch.no_grad():
        for inputs, labels, masks in test_loader:
            output = model.forward(inputs)
            y_pred.append(output)
    
    # 将所有预测结果转换为张量
    y_pred = torch.cat(y_pred, dim=0)  # 依据你的模型输出维度选择合适的dim参数
    y_shape = y_pred.shape
    if len(y_shape) > 2 and y_shape[-1]==1:  # 需要压缩维度
        y_pred = y_pred.squeeze()
    n_test = y_shape[0]

    y_pred_df = pd.DataFrame(y_pred.cpu())
    y_pred_df.columns = symbols
    y_pred_df.index = pd.to_datetime(dt[-n_test:])
    y_pred_df.index.name = 'datetime'

    # 将预测结果与原始数据合并
    y_pred_long = y_pred_df.reset_index().melt(id_vars=['datetime'])
    assert type(y_pred_long.iloc[0,1]) == str, "y_pred_long symbol列应该是字符串类型。检查数据格式。"
    y_pred_long.columns = ['datetime', 'symbol', 'value']
    full_df = full_df.collect().to_pandas()
    merge_df = full_df.merge(y_pred_long, on=['datetime', 'symbol'], how='inner')
    pl_df = pl.DataFrame(merge_df).lazy()

    hb = HtmlBuilder()
    x = 'value'
    y = 'la60'
    hb.add_title("Nerv ML Report")
    hb.add_plotly(daily_rel(pl_df, x))
    hb.add_plotly(pl_scatter(pl_df, x, y))
    hb.add_plotly(pdist(pl_df, x))
    hb.add_dataframe(analysis(pl_df, x, y), title="Summary of la60")

    hb.add_hr()
    hb.export(report_name)

    return pl_df

