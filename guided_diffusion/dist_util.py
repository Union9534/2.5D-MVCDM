"""
Helpers for training.
简化版本：移除了MPI和分布式依赖，支持单GPU训练。
"""

import io
import os
import torch as th


def setup_dist():
    """
    简化的单GPU设置
    """
    print("单GPU模式：跳过分布式初始化")
    pass


def dev():
    """
    获取设备
    """
    if th.cuda.is_available():
        # 如果有GPU，使用GPU 0
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    直接加载PyTorch文件
    """
    # 直接使用torch.load，不需要MPI通信
    return th.load(path, **kwargs)


def sync_params(params):
    """
    单进程不需要同步参数
    """
    pass


def _find_free_port():
    """
    单GPU不需要端口查找
    """
    return 29500
