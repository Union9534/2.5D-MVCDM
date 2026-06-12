"""
Train a diffusion model on images. 基于pytorch的扩散模型训练脚本
"""
import argparse
import os
import time
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from dataloader_scripts.load_pet_2_5D import LoadValData, load_data
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
def main():
    args = create_argparser().parse_args()
    args.in_channels = args.load_adj * 2 + 1 + args.out_channels    # 计算输入通道数
    start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # ��������
    dist_util.setup_dist()      # 设置分布式训练环境，支持多GPU训练
    logger.configure(dir=args.logdir)   # 配置日志系统，所有训练日志和输出将保存到指定的 logdir 目录

    logger.log("creating model and diffusion...")       # 创建模型和扩散过程
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.to(dist_util.dev())       # 模型移动到GPU/CPU设备
    # 直接使用CPU或第一个GPU
    device = dist_util.dev()
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # 创建时间步长采样器，控制训练时对不同噪声水平的采样策略

    logger.log("creating data loader...")
    # 加载训练数据
    train_dir = args.data_root
    data = load_data(args.batch_size, root_dir=train_dir, axis=args.train_axis, load_adj=args.load_adj)

    # 加载验证数据
    val_dir = os.path.join(args.data_root)
    val_data = LoadValData(root_dir=val_dir, axis=args.train_axis, load_adj=args.load_adj)

    logger.log("training...")       # 开始训练
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,                     # 微批量大小，适合显存不足时使用
        lr=args.lr,
        ema_rate=args.ema_rate,                         # 指数移动平均率，0.9999表示使用EMA平滑模型参数，提高稳定性
        log_interval=args.log_interval,                 # 日志记录间隔，每x步记录一次训练状态
        save_interval=args.save_interval,               # 模型保存间隔，每y步保存一次检查点
        resume_checkpoint=args.resume_checkpoint,       # 从之前保存的模型继续训练
        use_fp16=args.use_fp16,                         # 是否使用混合精度训练，使用FP16降低显存占用，加速训练
        fp16_scale_growth=args.fp16_scale_growth,       # FP16梯度缩放增长率，防止梯度下溢
        schedule_sampler=schedule_sampler,              # 控制训练时对不同噪声水平的采样策略
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,           # 学习率衰减步数，在多少步后将学习率降为0
        logdir=args.logdir,                           # 日志和检查点保存目录
    ).run_loop()

    end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # ��������
    print("������������{}>>>>>>>>>>>>>>>>������������{}".format(start_time, end_time))

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",     # 扩散时间步采样器，uniform为均匀采样所有时间步
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=30000,         # 学习率衰减，200000步内线性衰减到0（总训练步数）
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches 不用
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=30000,        # 模型保存间隔10000
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        train_axis="z",  # axis of model train on: x, y, or z
        load_adj=8,  # number of adjacent slices loaded as 2.5D condition input
        out_channels=1,
        logdir="models_7.3/checkpoint_z",  # save checkpoint and log in this folder
        data_root="./parihaka_save1/imdb_32_1.mat",
    )
    defaults.update(model_and_diffusion_defaults())    # model_and_diffusion_defaults()返回包含模型架构和扩散过程参数的字典，update将这些参数添加到现有的defaults字典中，合并参数
    parser = argparse.ArgumentParser()                 # 创建参数解析器对象
    add_dict_to_argparser(parser, defaults)            # 将字典中的所有参数添加到解析器中
    return parser                                      # 返回配置好的参数解析器，供主函数使用


if __name__ == "__main__":
    main()
