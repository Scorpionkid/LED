import os
import re
from torch.utils import data as data
import numpy as np
import torch
import glob
from os import path as osp
from tqdm import tqdm

from led.utils.registry import DATASET_REGISTRY
from led.data.raw_utils import pack_raw_bayer

@DATASET_REGISTRY.register()
class RAWGTDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.ratio_range = list(opt.get('ratio_range', (100, 300)))
        self.ratio_range[-1] = self.ratio_range[-1] + 1
        self.use_patches = opt.get('use_patches', True)
        self.load_in_mem = opt.get('load_in_mem', False)
        self.patch_id_max = opt.get('patch_id_max', 8)
        self.patch_tplt = opt.get('patch_tplt', '_s{:03}')

        # 新增：支持从data_pair_list中读取数据对信息
        self.data_pair_list = opt.get('data_pair_list', None)

        self.data_paths = []
        self.ratios = []
        self.iso_values = []

        if self.data_pair_list is not None:
            # 从文件列表中读取数据对
            with open(self.data_pair_list, 'r') as data_pair_list_file:
                pairs = data_pair_list_file.readlines()
                for pair in pairs:
                    parts = pair.split(' ')
                    if len(parts) < 2:
                        continue

                    lq, gt = parts[:2]
                    gt = gt.rstrip('\n')

                    # 提取曝光时间并计算比率
                    try:
                        shutter_lq = float(re.search(r'_(\d+(\.\d+)?)s\.', lq).group(1))
                        shutter_gt = float(re.search(r'_(\d+(\.\d+)?)s\.', gt).group(1))
                        ratio = min(shutter_gt / shutter_lq, 300)
                    except:
                        # 如果无法提取曝光时间，使用默认比率范围
                        ratio = torch.randint(self.ratio_range[0], self.ratio_range[1], (1,)).item()

                    # 提取ISO信息（如果有）
                    iso = None
                    if len(parts) > 2:
                        for part in parts[2:]:
                            if part.startswith('ISO'):
                                try:
                                    iso = int(part[3:])
                                except:
                                    pass
                                break

                    # 从GT文件路径中提取完整的文件名以供匹配
                    # 例如：从"./long/10003_00_10s.ARW"中提取"10003_00_10s"
                    basename = osp.basename(gt)
                    file_id = osp.splitext(basename)[0]  # 例如："10003_00_10s"

                    if self.use_patches:
                        # 在训练集目录中查找所有匹配的patches
                        # 格式可能是：*_00_10s_s001.npz
                        for i in range(1, self.patch_id_max + 1):
                            # 构建patch名称模式
                            patch_suffix = self.patch_tplt.format(i)

                            # 使用通配符查找匹配的文件
                            # 使用*来匹配所有前缀，但保留特定的曝光信息（00_10s部分）
                            exposure_part = re.search(r'_(\d+_\d+s)', file_id)
                            if exposure_part:
                                exposure_info = exposure_part.group(1)  # 例如："00_10s"

                                # 查找所有包含这个曝光信息和patch序号的文件
                                pattern = osp.join(self.root_folder, f"*_{exposure_info}{patch_suffix}")
                                if self.postfix:
                                    pattern += f".{self.postfix}"

                                matching_files = glob.glob(pattern)
                                for patch_path in matching_files:
                                    self.data_paths.append(patch_path)
                                    self.ratios.append(ratio)
                                    if iso is not None:
                                        self.iso_values.append(iso)
        else:
            # 原始行为：使用文件夹中的所有文件
            pattern = f'*.{self.postfix}' if self.postfix else '*'
            self.data_paths = glob.glob(osp.join(self.root_folder, pattern))

        if self.load_in_mem:
            self.datas = {
                data_path: self.depack_meta(data_path, self.postfix, True)
                for data_path in tqdm(self.data_paths, desc='loading data in mem...')
            }

    @staticmethod
    def depack_meta(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            import rawpy
            ## using rawpy
            raw = rawpy.imread(meta_path)
            raw_vis = raw.raw_image_visible.copy()
            raw_pattern = raw.raw_pattern
            wb = np.array(raw.camera_whitebalance, dtype='float32').copy()
            wb /= wb[1]
            ccm = np.array(raw.rgb_camera_matrix[:3, :3],
                           dtype='float32').copy()
            black_level = np.array(raw.black_level_per_channel,
                                   dtype='float32').reshape(4, 1, 1)
            white_level = np.array(raw.camera_white_level_per_channel,
                                   dtype='float32').reshape(4, 1, 1)
            im = pack_raw_bayer(raw_vis, raw_pattern).astype('float32')
            raw.close()
        else:
            raise NotImplementedError

        if to_tensor:
            im = torch.from_numpy(im).float().contiguous()
            black_level = torch.from_numpy(black_level).float().contiguous()
            white_level = torch.from_numpy(white_level).float().contiguous()
            wb = torch.from_numpy(wb).float().contiguous()
            ccm = torch.from_numpy(ccm).float().contiguous()

        return im, \
               black_level, white_level, \
               wb, ccm

    def randint(self, *range):
        return torch.randint(*range, size=(1,)).item()

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        # 使用存储的ratio（如果有），否则生成随机ratio
        ratio = self.ratios[index] if index < len(self.ratios) else self.randint(*self.ratio_range)
        # 使用存储的ISO（如果有）
        iso = self.iso_values[index] if index < len(self.iso_values) else None

        if not self.load_in_mem:
            im, black_level, white_level, wb, ccm = self.depack_meta(data_path, self.postfix, True)
        else:
            im, black_level, white_level, wb, ccm = self.datas[data_path]

        if self.opt['crop_size'] is not None:
            _, H, W = im.shape
            crop_size = self.opt['crop_size']
            assert crop_size < H and crop_size < W, f"裁剪尺寸 {crop_size} 超过图像尺寸 {H}x{W}"
            if self.opt['phase'] == 'train':
                h_start = self.randint(0, H - crop_size)
                w_start = self.randint(0, W - crop_size)
            else:
                # 中心裁剪
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            im_patch = im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            im_patch = im

        lq_im_patch = im_patch
        gt_im_patch = im_patch

        result = {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': ratio if isinstance(ratio, torch.Tensor) else torch.tensor(ratio).float(),
            'black_level': black_level,
            'white_level': white_level,
            'wb': wb,
            'ccm': ccm,
            'lq_path': data_path,
            'gt_path': data_path,
        }

        if iso is not None:
            result['iso'] = torch.tensor(iso).float()

        return result

    def __len__(self):
        return len(self.data_paths)