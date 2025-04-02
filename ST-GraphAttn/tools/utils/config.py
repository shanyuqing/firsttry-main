from pathlib import Path
import yaml
from easydict import EasyDict

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def merge_new_config(config, new_config):
    # base config 存放了一些公共的配置。例如 DATA_PATH
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():  # 递归添加新的配置
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


cfg = EasyDict()
# 设置为当前脚本所在目录的上一级目录的绝对路径。
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve() 