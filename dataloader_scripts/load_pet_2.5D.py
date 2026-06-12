"""
   --- ??????????????---
   --- ??????????????????????????---
"""
import os
import glob
import numpy as np
# from utility.utility import minmax_normalization
from torch.utils.data import Dataset
# import nibabel as nib  # ??????????????????????????NIFTI??DICOM??????????????
import scipy.io as sio
import h5py
from data_loader import SeisDataset
import random
from tqdm import tqdm
from skimage.transform import rescale
import torch
from torch.utils.data import DataLoader
import scipy.ndimage as ndimage  # ????????????????????????????????????


# ??pet??????????????????????
def normalize_ac(data, mask):
    non_zero_data = data[mask == True]
    non_zero_mean = np.mean(non_zero_data)
    data = data / non_zero_mean
    data = np.tanh(data / 5)  # ??????????????????????????
    return data


# ??pet??????????????????mask????????????????????????
def get_mask(nac, num_blurred=5, threshold=0.05):
    blurred_nac = np.copy(nac)
    for i in range(num_blurred):
        blurred_nac = ndimage.gaussian_filter(blurred_nac, sigma=1)  # ??????????1????????????????????????????????
    mask = np.where(nac > threshold, True, False)  # ??????????True????????False
    return mask


def data_load(data_dir, name='data'):
    try:
        data = sio.loadmat(data_dir)
        data = data[name]
    except:
        data = h5py.File(data_dir)
        data = data[name]
    return data


def miss_batch_trace_all(gt_data):
    miss_position = np.ones_like(gt_data)

    for i in range(0, miss_position.shape[0]):
        miss_type = random.randint(0, 2)
        miss_position_item = miss_position[i, :, :, :, :]
        shape = miss_position_item.shape
        # ????????
        if miss_type in [0, 1]:
            miss_position_item = np.reshape(miss_position_item, [shape[0], shape[1], shape[2] * shape[3]])
            # b c t x y
            all_len = shape[2] * shape[3]
            # old
            # missing_coefficient = round(random.uniform(0.5, 0.7), 2)
            missing_coefficient = round(random.uniform(0.3, 0.7), 2)
            miss_id = np.array(
                random.sample(range(1, all_len - 1), np.int32(np.around(all_len * missing_coefficient))))

            for xy in miss_id:
                miss_position_item[:, :, xy] = 0
            miss_position_item = miss_position_item.reshape((shape[0], shape[1], shape[2], shape[3]))
        # ????????
        elif miss_type in [2]:
            start = random.randint(2, 28)
            end = random.randint(start, 28)
            # start = random.randint(4, 24)
            # end = random.randint(start, 24)
            # length = random.randint(7, 13)
            # start = random.randint(2, 28-length+1)
            # end = start + length
            miss_position_item[:, :, :, start:end] = 0
        # ????????
        elif miss_type in [3]:
            missing_coefficient = 0.3
            # width = 3
            interval = np.int32(np.around(1.0 / missing_coefficient))
            # for x in range(np.int32(interval / 2), np.shape(miss_position_item)[1], interval):
            #     miss_position_item[:, :, x, :] = 0
            for y in range(np.int32(interval / 2), np.shape(miss_position_item)[3], interval):
                miss_position_item[:, :, :, y] = 0
        miss_position[i, :, :, :, :] = miss_position_item
    return miss_position


# pytorch??????????????????????????????pet????????
class LoadPetSlices(Dataset):
    def __init__(self, root_dir=r"", axis="z", load_adj=12, seed=1, out_size=32) -> None:
        super().__init__()
        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir

        complete_data = data_load(root_dir, 'inputs_train')
        complete_data = np.array(complete_data, dtype=np.float32)
        print(complete_data.shape)  # ??3440??1??32??32??32??
        complete_data = np.transpose(complete_data, [4, 3, 0, 1, 2])
        # complete_data = complete_data[1:130, :, :, :, :]
        print(complete_data.shape)
        # num_train = complete_data.shape[0]
        # complete_data = complete_data[0:((num_train // 16) * 16), :, :, :, :]
        # print(complete_data.shape)
        complete_data = SeisDataset(complete_data)
        train_loader = torch.utils.data.DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
        # nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
        # ac_path = os.path.join(self.root_dir, '100AC', self.ids[i] + "100_AC.nii")
        # nac = nib.load(nac_path).get_fdata()
        # ac = nib.load(ac_path).get_fdata()
        # mask = get_mask(nac)
        self.com_data = []
        self.miss_data = []
        loop = tqdm(train_loader, desc=None)
        for batch_idx1, (gt_data) in enumerate(loop, 0):
            mask = miss_batch_trace_all(gt_data)
            mask = torch.from_numpy(mask).float()
            miss_data = torch.multiply(gt_data, mask)
            self.com_data.append(gt_data)
            self.miss_data.append(miss_data)

    def __len__(self):
        return len(self.com_data)

    # ??3D??????????????????2.5D????????
    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z=0):
        # ????????????
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)  # ????????????????????????????????????????????????????????????????????
        # ????????????
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1, 0, 2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2, 0, 1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)  # ??????0????????????????????????????
        comb_slices[fill_start:fill_start + slices.shape[0]] = slices  # ????????????

        return comb_slices  # ????2.5D????

    # ??NIFTI????????????
    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)  # ??????????
        shape = (self.out_size, self.out_size, max(data.shape[2], self.out_size))  # ??????????????????????????z????????out_size
        new_data = np.zeros(shape)
        new_data[:, :, :data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2,
                                         (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2,
                                         :]  # ????????????data??????0????new_data
        return torch.from_numpy(new_data)

    def random_flip(self, image, label, flip_axis, p=0.3):
        for i in flip_axis:  # ????????????????
            if np.random.rand() < p:  # ????0-1????????????p??????????
                image = torch.flip(image, [i])  # ??????????????????
                label = torch.flip(label, [i])
        return image, label

    # ????????????????????????????????????????????????????????
    def __getitem__(self, idx):
        ac = self.com_data[idx]  # ????????AC??????????????
        nac = self.miss_data[idx]  # ????????NAC??????????????
        ac = ac.squeeze()
        nac = nac.squeeze()
        # print(nac.shape)

        z_size = ac.shape[2]
        if self.axis != "z" and z_size > self.out_size:
            start_z = random.randint(0, z_size - self.out_size)
            central_slice_idx = random.randint(0, self.out_size - 1)
        else:
            start_z = 0
            central_slice_idx = random.randint(0, z_size - 1)

        # ????2.5D????
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)

        # ??????????2D????
        if self.axis == "x":
            ac = ac[central_slice_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            ac = ac[:, central_slice_idx, start_z:start_z + self.out_size]
        elif self.axis == "z":
            ac = ac[:, :, central_slice_idx]
        ac = ac.unsqueeze(0)

        # ????????????????????[2.5D????(17*192*192)??2D????(1*192*192??]
        nac, ac = self.random_flip(nac, ac, flip_axis=[0, 1, 2])
        return ac, nac


def load_data(batch_size, root_dir=r"", axis="z", load_adj=12, seed=1, out_size=32):
    dataset = LoadPetSlices(root_dir=root_dir, axis=axis, load_adj=load_adj, seed=seed, out_size=out_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        yield from loader


# ????????????
class LoadTestData():
    def __init__(self, root_dir=r"", axis="z", load_adj=8, seed=1, out_size=192) -> None:

        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir
        # self.file_names = os.listdir(os.path.join(root_dir, '5NAC'))
        # self.ids = [i[:len(i) - 9] for i in self.file_names]

        # self.len = len(self.ids)
        self.uncomplete_data = []
        self.original_z = []

        uncomplete = data_load(root_dir, 'missed_data')  # jia zai 1 miss_data
        uncomplete = np.array(uncomplete, dtype=np.float32)
        self.uncomplete_data = [uncomplete]

        # ????NAC??????????x,y,z??
        # for i in tqdm(range(len(self.ids))):
        #     nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
        #     nac = nib.load(nac_path).get_fdata()
        #     mask = get_mask(nac)
        #     self.original_z.append(nac.shape[2])
        #     nac = self.process_nii_file(nac, mask)
        #     self.nac_data.append(nac)
        #
        self.current_idx = 0

    def __len__(self):
        return len(self.uncomplete_data)

    def set_current_index(self, idx):
        """????????????????????????????????????"""
        if idx < 0 or idx >= len(self.uncomplete_data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.uncomplete_data) - 1}]")
        self.current_idx = idx

    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z=0):
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)
        image_3D = torch.from_numpy(image_3D).float()
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1, 0, 2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2, 0, 1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)

        comb_slices[fill_start:fill_start + slices.shape[0]] = slices

        return comb_slices

    # ??????????????Z??????????????????
    # def get_zsize(self, idx=None):
    #     if idx == None:
    #         idx = self.idx
    #     return self.uncomplete_data[idx].shape[2]

    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)
        shape = (self.out_size, self.out_size, max(data.shape[2], self.out_size))
        new_data = np.zeros(shape)
        new_data[:, :, :data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2,
                                         (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2, :]
        return torch.from_numpy(new_data)

    def get_slices(self, central_slice_idx, idx=None, start_z=0):
        if idx is None:
            idx = self.current_idx
        else:
            if idx < 0 or idx >= len(self.uncomplete_data):
                raise IndexError(f"Index {idx} out of range")
        nac = self.uncomplete_data[idx]
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)
        return nac
    #
    # def get_target(self, idx=None):
    #     if idx == None:
    #         idx = self.idx
    #     return self.ac[idx]
    #
    # def get_name(self, idx=None):
    #     if idx == None:
    #         idx = self.idx
    #     return self.idx[idx]
    #
    # def get_original_z(self, idx=None):
    #     if idx == None:
    #         idx = self.idx
    #     return self.original_z[idx]
    #
    # def normalize(self, data, idx=None):
    #     if idx == None:
    #         idx = self.idx
    #     nac_mean = self.means[idx]
    #     data /= nac_mean
    #     data = np.tanh(data / 5)
    #     return data


# ????????????
class LoadValData():
    def __init__(self, root_dir=r"", axis="y", load_adj=8, seed=1, out_size=32) -> None:

        assert axis in ["x", "y", "z"]
        self.out_size = out_size
        self.axis = axis
        self.load_adj = load_adj
        random.seed(seed)
        self.root_dir = root_dir

        complete_data = data_load(root_dir, 'inputs_train')
        complete_data = np.array(complete_data, dtype=np.float32)
        print(complete_data.shape)  # ??3440??1??32??32??32??
        # num_train = complete_data.shape[0]
        # complete_data = complete_data[0:((num_train // 16) * 16), :, :, :, :]
        # print(complete_data.shape)
        complete_data = SeisDataset(complete_data)
        train_loader = torch.utils.data.DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
        # nac_path = os.path.join(self.root_dir, '5NAC', self.ids[i] + "5_NAC.nii")
        # ac_path = os.path.join(self.root_dir, '100AC', self.ids[i] + "100_AC.nii")
        # nac = nib.load(nac_path).get_fdata()
        # ac = nib.load(ac_path).get_fdata()
        # mask = get_mask(nac)
        self.com_data = []
        self.miss_data = []
        loop = tqdm(train_loader, desc=None)
        for batch_idx1, (gt_data) in enumerate(loop, 0):
            mask = miss_batch_trace_all(gt_data)
            mask = torch.from_numpy(mask).float()
            miss_data = torch.multiply(gt_data, mask)
            self.com_data.append(gt_data)
            self.miss_data.append(miss_data)

    def __len__(self):
        return len(self.com_data)

    def convert_3d_to_25d(self, image_3D, central_slice_idx, start_z=0):
        start_idx = max(central_slice_idx - self.load_adj, 0)
        end_idx = central_slice_idx + self.load_adj + 1
        fill_start = max(self.load_adj - central_slice_idx, 0)
        if self.axis == "x":
            slices = image_3D[start_idx:end_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            slices = image_3D[:, start_idx:end_idx, start_z:start_z + self.out_size].permute(1, 0, 2)
        elif self.axis == "z":
            slices = image_3D[:, :, start_idx:end_idx].permute(2, 0, 1)
        else:
            raise ValueError("Invalid axis: choose from 'x', 'y', or 'z'")

        comb_slices = torch.zeros(self.load_adj * 2 + 1, self.out_size, self.out_size)
        comb_slices[fill_start:fill_start + slices.shape[0]] = slices

        return comb_slices

    def get_zsize(self, idx=None):
        if idx == None:
            idx = self.idx
        return self.nac_data[idx].shape[2]

    def process_nii_file(self, data, mask):
        data = normalize_ac(data, mask)
        shape = (self.out_size, self.out_size, max(data.shape[2], self.out_size))
        new_data = np.zeros(shape)
        new_data[:, :, :data.shape[2]] = data[(data.shape[0] - shape[0]) // 2: (data.shape[0] + shape[0]) // 2,
                                         (data.shape[1] - shape[1]) // 2: (data.shape[1] + shape[1]) // 2, :]
        return torch.from_numpy(new_data)

    def get_slices(self, central_slice_idx, idx=None, start_z=0):
        if idx == None:
            idx = self.idx
        nac = self.miss_data[idx]
        nac = nac.squeeze()
        nac = self.convert_3d_to_25d(nac, central_slice_idx, start_z)
        return nac

    def get_target_slices(self, central_slice_idx, idx=None, start_z=0):
        if idx == None:
            idx = self.idx
        ac = self.com_data[idx]
        if self.axis == "x":
            ac = ac[central_slice_idx, :, start_z:start_z + self.out_size]
        elif self.axis == "y":
            ac = ac[:, central_slice_idx, start_z:start_z + self.out_size]
        elif self.axis == "z":
            ac = ac[:, :, central_slice_idx + start_z]
        ac = ac.unsqueeze(0)
        return ac

    def get_target(self, idx=None):
        if idx == None:
            idx = self.idx
        return self.ac[idx]

    def get_name(self, idx=None):
        if idx == None:
            idx = self.idx
        return self.ids[idx]

    def get_original_z(self, idx=None):
        if idx == None:
            idx = self.idx
        return self.original_z[idx]
