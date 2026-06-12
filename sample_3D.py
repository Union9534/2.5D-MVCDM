"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
加载多个预训练的扩散模型，对测试数据进行采样生成3D医学图像，并保存为NIFTI格式文件
"""
import numpy as np
import torch as th
import torch.distributed as dist
import argparse
import os
import time
import torch.nn as nn
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataloader_scripts.load_pet_2_5D import LoadTestData
# import nibabel as nib
import scipy.io as sio
import h5py

def main():
    args = create_argparser().parse_args()
    start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # ��������
    args.in_channels = args.load_adj * 2 + 1 + args.out_channels

    dist_util.setup_dist()
    logger.configure()
    logger.log("loading model and diffusion...")
    models = []
    diffusion = None
    # 加载多个模型（沿不同轴训练）
    for i in args.model_axis:
        model_path = os.path.join(f"{args.model_root}_{i}", f"model030000.pt")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model.requires_grad_(False)
        models.append(model)

    def data_load(data_dir, name='data'):
        try:
            data = sio.loadmat(data_dir)
            data = data[name]
        except:
            data = h5py.File(data_dir)
            data = data[name]
        return data

    logger.log("sampling...")
    # test_dir = os.path.join(args.data_root, "test")
    test_dir = './dataset/test_parihaka/produce_mask_data31_im80_parihaka.mat'
    test_input = LoadTestData(root_dir=test_dir, load_adj=args.load_adj)

    # 对每个测试样本进行采样
    for idx in range(len(test_input)):
        print(f"idx: {idx} / {len(test_input)}")
        whole_image = None          # 储存生成的完整3D图像
        sample_fn = diffusion.p_sample_loop     # 将扩散模型的采样函数赋值给sample_fn
        model_kwargs = {}           # 创建一个空字典，用于传递给模型的前向传播参数
        # test_input.idx = idx        # 设置测试数据加载器的当前索引
        shape = (args.image_size, args.image_size, args.image_size)              # 定义生成图像的3D形状
        # shape = (128, 128, 128)
        prior_data = data_load(test_dir, 'missed_data')
        prior_numpy = np.array(prior_data, dtype=np.float32)
        comb_img = np.zeros(shape)                              # 初始化一个全零的NumPy数组，用于累积多次采样的结果
        for i in range(args.sample_num):                        # 对每个测试样本进行sample_num次采样
            prior = th.zeros(shape).to(dist_util.dev())         # 创建与目标形状相同的全零张量，并转移到GPU
            prior[:, :, :] = th.from_numpy(prior_numpy).to(dist_util.dev())          # 将先验数据填充到前get_original_z()个切片中，不是每个切片都有先验信息
            noisy_priors = []           # 生成多个噪声先验作为扩散模型采样的起始点
            for n in range(args.avg_start_number): 
                if args.prior_start_t != None:
                    noisy_priors.append(diffusion.q_sample(prior, th.tensor(args.prior_start_t).to(dist_util.dev())))   # 扩散模型的前向过程，在指定时间步prior_start_t对先验图像添加噪声，生成部分破坏的先验图像
                else:
                    noisy_priors.append(th.randn(shape).to(dist_util.dev()))            # 标准正态分布生成完全随机的噪声
            # 开始采样
            whole_image = sample_fn(
                models,             # 多个预训练模型
                args.model_axis,    # 模型对应的轴向
                test_input,         # 测试数据（提供条件信息）
                shape,
                args.batch_size,
                args.prior_start_t,
                noise=noisy_priors,     # 使用生成的噪声先验作为起点
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            whole_image = whole_image.cpu().numpy()     # 将GPU张量转移到CPU并转换为NumPy数组
            comb_img += whole_image                     # 将当前采样结果累加到 comb_img 中用于后续平均
            # whole_image = whole_image[:, :, :test_input.get_original_z()]       # 裁剪到原始z轴尺寸（移除填充部分）
            # whole_image = nib.Nifti1Image(whole_image, affine=np.eye(4))        # 转换为NIfTI图像格式

            axis = "".join(args.model_axis)             # 将模型轴向列表转换为字符串，args.model_axis = ["x", "y", "z"]，则 axis = "xyz"，在文件中标识使用了哪些轴向的模型

            # Save individual sampled volume ，保存单个采样的生成结果
            output_dir = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_ave_first_ddpm_full_single")
            os.makedirs(output_dir, exist_ok=True)

            # save_path = os.path.join(output_dir, f"{test_input.get_name(idx)}pred_{i}.nii")
            # nib.save(whole_image, save_path)
            sio.savemat(os.path.join(output_dir, f"pred_{i}.mat"),
                        {'pred': whole_image})

            # Save averaged combined volume ，保存多次采样的平均结果
            comb_img /= args.sample_num
            # comb_img = comb_img[:, :, :test_input.get_original_z()]
            comb_img[comb_img < 0] = 0
            # comb_img = nib.Nifti1Image(comb_img, affine=np.eye(4))

            output_dir_comb = os.path.join(args.save_root, f"adj{args.load_adj}_models_{axis}", f"noise_{args.avg_start_number}_priort_{args.prior_start_t}_comb")
            os.makedirs(output_dir_comb, exist_ok=True)

            # save_path_comb = os.path.join(output_dir_comb, f"{test_input.get_name(idx)}pred.nii")
            # nib.save(comb_img, save_path_comb)
            sio.savemat(os.path.join(output_dir_comb, f"pred.mat"),
                        {'pred_avg': comb_img})

    end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # ��������
    print("������������{}>>>>>>>>>>>>>>>>������������{}".format(start_time, end_time))

def create_argparser():
    defaults = dict(
        clip_denoised=True,         # 是否对去噪后的图像进行值域裁剪
        batch_size=1,
        use_ddim=False,             # 是否使用DDIM采样方法
        out_channels=1,
        model_root="models_2/checkpoint",        # 预训练模型文件的存储根目录
        model_axis=["x", "y", "z"],
        # model_axis=["y", "z"],
        prior_start_t=200,          # 从扩散过程的第200个时间步开始进行反向去噪过程， range from 1 to 999 for starting noise level add to prior, None for no prior
        load_adj=8,
        avg_start_number=2,         # 从多少个不同的噪声点开始采样并平均结果，增加该值可以提高结果的稳定性（提高质量）
        sample_num=1,               # 每个样本的采样次数
        save_root="results_models_2_im80",           # 生成结果的保存根目录
        load_prior_root="cGAN_prior",       # 先验图像文件的存储目录
        data_root="NAC_data",               # 原始数据的存储根目录
    )
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
