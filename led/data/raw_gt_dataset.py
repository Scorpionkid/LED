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

        # New：Support reading data pair information from the data_pair_list.
        self.data_pair_list = opt.get('data_pair_list', None)

        self.data_paths = []
        self.ratios = []
        self.iso_values = []

        if self.data_pair_list is not None:
            with open(self.data_pair_list, 'r') as data_pair_list_file:
                pairs = data_pair_list_file.readlines()
                for pair in pairs:
                    parts = pair.split(' ')
                    if len(parts) < 2:
                        continue

                    lq, gt = parts[:2]
                    gt = gt.rstrip('\n')

                    try:
                        shutter_lq = float(re.search(r'_(\d+(\.\d+)?)s\.', lq).group(1))
                        shutter_gt = float(re.search(r'_(\d+(\.\d+)?)s\.', gt).group(1))
                        ratio = min(shutter_gt / shutter_lq, 300)
                    except:
                        # If the exposure time cannot be extracted, use the default ratio range.
                        ratio = torch.randint(self.ratio_range[0], self.ratio_range[1], (1,)).item()

                    # ISO extraction
                    iso = None
                    if len(parts) > 2:
                        for part in parts[2:]:
                            if part.startswith('ISO'):
                                try:
                                    iso = int(part[3:])
                                except:
                                    pass
                                break

                    # e.g."./long/10003_00_10s.ARW"->"10003_00_10s"
                    basename = osp.basename(gt)
                    file_id = osp.splitext(basename)[0]  # e.g.："10003_00_10s"

                    if self.use_patches:
                        # Find all matching patches in the training set directory.
                        # e.g.：*_00_10s_s001.npz
                        for i in range(1, self.patch_id_max + 1):
                            # patch name pattern
                            patch_suffix = self.patch_tplt.format(i)
                            file_ext = ".ARW"

                            pattern = osp.join(self.root_folder, f"{file_id}{patch_suffix}{file_ext}")
                            if self.postfix:
                                pattern += f".{self.postfix}"

                            matching_files = glob.glob(pattern)
                            for patch_path in matching_files:
                                self.data_paths.append(patch_path)
                                self.ratios.append(ratio)
                                if iso is not None:
                                    self.iso_values.append(iso)
        else:
            # Original behavior: Use all files in the folder.
            pattern = f'*.{self.postfix}' if self.postfix else '*'
            self.data_paths = glob.glob(osp.join(self.root_folder, pattern))

        self.load_in_mem = opt.get('load_in_mem', False)
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
        ratio = self.ratios[index] if index < len(self.ratios) else self.randint(*self.ratio_range)
        iso = self.iso_values[index] if index < len(self.iso_values) else None

        if not self.load_in_mem:
            im, black_level, white_level, wb, ccm = self.depack_meta(data_path, self.postfix, True)
        else:
            im, black_level, white_level, wb, ccm = self.datas[data_path]

        if self.opt['crop_size'] is not None:
            _, H, W = im.shape
            crop_size = self.opt['crop_size']
            assert crop_size < H and crop_size < W, f"crop size {crop_size} > img size {H}×{W}"
            if self.opt['phase'] == 'train':
                h_start = self.randint(0, H - crop_size)
                w_start = self.randint(0, W - crop_size)
            else:
                # center crop
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