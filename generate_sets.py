import numpy as np
from scipy.io import savemat, loadmat


def extract_2d_blocks(data, block_size=(64, 64), stride=(32, 32)):
    t, h, w = data.shape
    blocks = []

    # 沿t-h平面截取块（固定每个w位置）
    for w_idx in range(w):
        for t_start in range(0, t - block_size[0] + 1, stride[0]):
            for h_start in range(0, h - block_size[1] + 1, stride[1]):
                block = data[t_start:t_start + block_size[0], h_start:h_start + block_size[1], w_idx]
                blocks.append(block[np.newaxis, ...])  # 添加通道维度

    # 沿t-w平面截取块（固定每个h位置）
    for h_idx in range(h):
        for t_start in range(0, t - block_size[0] + 1, stride[0]):
            for w_start in range(0, w - block_size[1] + 1, stride[1]):
                block = data[t_start:t_start + block_size[0], h_idx, w_start:w_start + block_size[1]]
                blocks.append(block[np.newaxis, ...])  # 添加通道维度

    # 合并所有块并打乱顺序
    all_blocks = np.vstack(blocks)
    # np.random.shuffle(all_blocks)
    all_blocks = all_blocks[np.newaxis, ...]
    all_blocks = np.transpose(all_blocks, (1,0,2,3))
    return all_blocks


def extract_3d_blocks(data, block_size=(64, 64, 64), stride=(32, 32, 32)):
    t, h, w = data.shape
    t_block, h_block, w_block = block_size
    t_stride, h_stride, w_stride = stride

    blocks = []

    # 遍历所有可能的起始位置
    for t_start in range(0, t - t_block + 1, t_stride):
        for h_start in range(0, h - h_block + 1, h_stride):
            for w_start in range(0, w - w_block + 1, w_stride):
                # 截取3D块
                block = data[
                        t_start:t_start + t_block,
                        h_start:h_start + h_block,
                        w_start:w_start + w_block
                        ]
                # 添加通道维度 (1, t_block, h_block, w_block)
                block = block[np.newaxis, ...]
                blocks.append(block)

    # 合并所有块为 (num_blocks, 1, t_block, h_block, w_block)
    all_blocks = np.concatenate(blocks, axis=0)

    # 随机打乱块顺序
    np.random.shuffle(all_blocks)
    all_blocks = all_blocks[np.newaxis, ...]
    all_blocks = np.transpose(all_blocks, (1,0,2,3,4))
    return all_blocks


if __name__ == "__main__":
    # 加载数据
    data_name = 'kerry_new'
    data_path = f'./datasets/{data_name}/all.mat'
    data = loadmat(data_path)
    data = data["data"]
    data = np.array(data, dtype=np.float32)
    # data = data[:,:, 0:384]
    # data = data[:,0:512, :]
    print(data.shape)

    # # 截取2D训练集
    result = extract_2d_blocks(data, block_size=(32, 32), stride=(32, 32))
    print(result.shape)
    # savemat(f'./datasets/{data_name}/train/2d/data.mat', {'data': result})

    # # 截取3D训练集
    result = extract_3d_blocks(data, block_size=(32, 32, 32), stride=(16, 16, 16))
    print(result.shape)
    # savemat(f'./datasets/{data_name}/train/3d/data.mat', {'data': result})

    # # 截取少量3D迁移训练集
    data = data[:,:,256:384]
    result = extract_3d_blocks(data, block_size=(32, 32, 32), stride=(16, 16, 16))
    print(result.shape)
    #
    # savemat(f'./datasets/{data_name}/train/3d_transfer/data.mat', {'data': result})