import re
from torch.utils import data as data
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm
import scipy.io as sio
import h5py
import os
import glob
from led.utils.registry import DATASET_REGISTRY
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.data_preparation.SIDD.Sub_lmdb.metadata_utils import extract_metadata_from_mat
from scripts.data_preparation.SIDD.SIDD_raw_Small_only.mat_to_npy import pack_raw_bayer as SIDD_pack_raw_bayer
from led.data.raw_utils import pack_raw_bayer

@DATASET_REGISTRY.register()
class PairedRAWDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.which_meta = opt.get('which_meta', 'gt')
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.use_patches = opt.get('use_patches', False)
        self.load_in_mem = opt.get('load_in_mem', False)
        if self.use_patches:
            assert self.load_in_mem == True
            self.patch_id_max = opt.get('patch_id_max', 8)
            self.patch_tplt = opt.get('patch_tplt', '_s{:03}')
        #     self.patch_tplt_re = opt.get('patch_tplt_re', '_s\d{3}')
        ## previous: only support npz with metadata
        # assert self.postfix == 'npz'

        self.lq_paths, self.gt_paths, self.ratios = [], [], []
        self.iso_values = []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                parts = pair.split(' ')
                lq, gt = parts[:2]
                gt = gt.rstrip('\n')
                shutter_lq = float(re.search(r'_(\d+(\.\d+)?)s.', lq).group(0)[1:-2])
                shutter_gt = float(re.search(r'_(\d+(\.\d+)?)s.', gt).group(0)[1:-2])
                ratio = min(shutter_gt / shutter_lq, 300)

                # iso extraction
                #TODO 非空检查
                if len(parts) > 2:
                    for part in parts[2:]:
                        if part.startswith('ISO'):
                            iso = int(part[3:])
                            self.iso_values.append(iso)
                            break

                if not self.use_patches:
                    self.lq_paths.append(osp.join(self.root_folder, lq))
                    self.gt_paths.append(osp.join(self.root_folder, gt))
                    self.ratios.append(ratio)
                else:
                    for i in range(1, 1 + self.patch_id_max):
                        self.lq_paths.append(osp.join(self.root_folder, self.insert_patch_id(lq, i, self.patch_tplt)))
                        self.gt_paths.append(osp.join(self.root_folder, self.insert_patch_id(gt, i, self.patch_tplt)))
                        self.ratios.append(ratio)
        if self.load_in_mem:
            get_data_path = lambda x: '.'.join([x, self.postfix]) if self.postfix is not None and self.postfix not in ['mat', 'MAT'] else x
            self.lqs = {
                get_data_path(data_path): \
                    self.depack_meta(get_data_path(data_path), self.postfix)
                for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
            }
            self.gts = {
                get_data_path(data_path): \
                    self.depack_meta(get_data_path(data_path), self.postfix)
                for data_path in tqdm(set(self.gt_paths), desc='load gt metas in mem...')
            }

    @staticmethod
    def insert_patch_id(path, patch_id, tplt='_s{:03}'):
        exts = path.split('.')
        base = exts.pop(0)
        while exts and exts[0] not in ['ARW', 'MAT']:
            base += '.' + exts.pop(0)
        base = base + tplt.format(patch_id)
        return base + '.' + '.'.join(exts)

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
        elif postfix in ['mat', 'MAT']:
            try:
                try:
                    mat_data = sio.loadmat(meta_path)
                except:
                    with h5py.File(meta_path, 'r') as f:
                        raw_data = None
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 2:
                                raw_data = np.array(f[key])
                                break
                        if raw_data is None:
                            raise ValueError(f"在{meta_path}中找不到RAW数据")

                if 'mat_data' in locals():
                    raw_data = None
                    if 'x' in mat_data:
                        raw_data = mat_data['x']
                    elif 'raw' in mat_data:
                        raw_data = mat_data['raw']
                    else:
                        for key in mat_data.keys():
                            if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                                raw_data = mat_data[key]
                                break
                    if raw_data is None:
                        raise ValueError(f"在{meta_path}中找不到RAW数据")

                camera_model = None
                path_parts = meta_path.split(os.path.sep)
                for part in path_parts:
                    if part in ['GP', 'IP', 'S6', 'N6', 'G4']:
                        camera_model = part
                        break

                cfa_pattern = 'rggb'

                metadata_file = None

                filename = os.path.basename(meta_path)
                match = re.match(r'(\d{5})_', filename)
                if match:
                    scene_instance = match.group(1).lstrip('0')
                    if not scene_instance:
                        scene_instance = '0'

                    sidd_data_path = '/media/HDD3/personal_files/zhouzhou/VideoDenoisingDatasets/SIDD_Small_Raw_Only/Data'

                    pattern = f"{scene_instance:0>4}_*_{camera_model}_*"
                    potential_dirs = glob.glob(os.path.join(sidd_data_path, pattern))

                    if potential_dirs:
                        for scene_dir in potential_dirs:
                            metadata_files = glob.glob(os.path.join(scene_dir, '*METADATA*.MAT'))
                            if metadata_files:
                                metadata_file = metadata_files[0]
                                break

                if metadata_file and os.path.exists(metadata_file):
                    try:
                        metadata = extract_metadata_from_mat(metadata_file)
                        if 'wb' in metadata:
                            wb = np.array(metadata['wb'], dtype='float32')
                        else:
                            wb = np.array([2.0, 1.0, 1.5, 1.0], dtype='float32')

                        if 'ccm' in metadata:
                            ccm = np.array(metadata['ccm'], dtype='float32')
                        else:
                            ccm = np.eye(3, dtype='float32')

                        if 'black_level' in metadata:
                            black_level = np.array(metadata['black_level'], dtype='float32')

                        if 'white_level' in metadata:
                            white_level = np.array(metadata['white_level'], dtype='float32')

                        if 'cfa_pattern' in metadata:
                            cfa_pattern = metadata['cfa_pattern']
                    except Exception as e:
                        print(f"无法从{metadata_file}提取元数据: {e}")
                        wb = np.array([2.0, 1.0, 1.5, 1.0], dtype='float32')
                        ccm = np.eye(3, dtype='float32')
                else:
                    wb = np.array([2.0, 1.0, 1.5, 1.0], dtype='float32')
                    ccm = np.eye(3, dtype='float32')

                if np.issubdtype(raw_data.dtype, np.integer):
                    raw_data = raw_data.astype('float32')

                if raw_data.max() <= 1.0:
                    black_level = metadata.get('black_level', np.zeros((4,1,1)))
                    if isinstance(black_level, np.ndarray) and black_level.size > 0:
                        black_level = black_level.reshape(-1)[0]
                    else:
                        black_level = 0

                    white_level = metadata.get('white_level', np.ones((4,1,1)) * 1023)
                    if isinstance(white_level, np.ndarray) and white_level.size > 0:
                        white_level = white_level.reshape(-1)[0]
                    else:
                        white_level = 1023

                    raw_data = raw_data * (white_level - black_level) + black_level

                # 打包为4通道
                im = SIDD_pack_raw_bayer(raw_data, cfa_pattern).astype(np.float32)

            except Exception as e:
                print(f"读取MAT文件{meta_path}时出错: {e}")
                # 设置默认值
                im = np.zeros((4, 128, 128), dtype='float32')
                black_level = np.zeros((4, 1, 1), dtype='float32')
                white_level = np.ones((4, 1, 1), dtype='float32')
                wb = np.array([2.0, 1.0, 1.5, 1.0], dtype='float32')
                ccm = np.eye(3, dtype='float32')
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
            raise NotImplementedError(f"不支持的文件后缀: {postfix}")

        if to_tensor:
            im = torch.from_numpy(np.asarray(im, dtype=np.float32)).contiguous()
            black_level = torch.from_numpy(np.asarray(black_level, dtype=np.float32)).contiguous()
            if black_level.dim() == 1:
                black_level = black_level.view(-1, 1, 1)
            elif black_level.dim() == 2:
                black_level = black_level.unsqueeze(-1)

            white_level = torch.from_numpy(np.asarray(white_level, dtype=np.float32)).contiguous()
            if white_level.dim() == 1:
                white_level = white_level.view(-1, 1, 1)
            elif white_level.dim() == 2:
                white_level = white_level.unsqueeze(-1)

            wb = torch.from_numpy(np.asarray(wb, dtype=np.float32)).contiguous()
            ccm = torch.from_numpy(np.asarray(ccm, dtype=np.float32)).contiguous()

        return (im - black_level) / (white_level - black_level), \
                wb, ccm

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.ratios[index]
        iso = self.iso_values[index]

        if self.postfix is not None and self.postfix not in ['MAT', 'mat']:
            lq_path = '.'.join([lq_path, self.postfix])
            gt_path = '.'.join([gt_path, self.postfix])

        # 检查并修改 lq_path 的后缀
        if lq_path.endswith('.MAT.npz'):
            lq_path = lq_path.replace('.MAT.npz', '.npz')

        # 检查并修改 gt_path 的后缀
        if gt_path.endswith('.MAT.npz'):
            gt_path = gt_path.replace('.MAT.npz', '.npz')

        if not self.load_in_mem:
            lq_im, lq_wb, lq_ccm = self.depack_meta(lq_path, self.postfix)
            gt_im, gt_wb, gt_ccm = self.depack_meta(gt_path, self.postfix)
        else:
            lq_im, lq_wb, lq_ccm = self.lqs[lq_path]
            gt_im, gt_wb, gt_ccm = self.gts[gt_path]

        ### augment
        ## crop
        if self.opt['crop_size'] is not None:
            _, H, W = lq_im.shape
            crop_size = self.opt['crop_size']
            assert crop_size <= H and crop_size <= W
            if self.opt['phase'] == 'train':
                h_start = torch.randint(0, H - crop_size, (1,)).item()
                w_start = torch.randint(0, W - crop_size, (1,)).item()
            else:
                # center crop
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            lq_im_patch = lq_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
            gt_im_patch = gt_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            lq_im_patch = lq_im
            gt_im_patch = gt_im
        ## flip + rotate
        if self.opt['phase'] == 'train':
            hflip = self.opt['use_hflip'] and torch.rand((1,)).item() < 0.5
            vflip = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            rot90 = self.opt['use_rot'] and torch.rand((1,)).item() < 0.5
            if hflip:
                lq_im_patch = torch.flip(lq_im_patch, (2,))
                gt_im_patch = torch.flip(gt_im_patch, (2,))
            if vflip:
                lq_im_patch = torch.flip(lq_im_patch, (1,))
                gt_im_patch = torch.flip(gt_im_patch, (1,))
            if rot90:
                lq_im_patch = torch.permute(lq_im_patch, (0, 2, 1))
                gt_im_patch = torch.permute(gt_im_patch, (0, 2, 1))

            if self.opt.get('ratio_aug') is not None:
                ratio_range = self.opt['ratio_aug']
                rand_ratio = torch.rand((1,)).item() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
                ## TODO: maybe there are some over-exposed?
                gt_im_patch = gt_im_patch / ratio * rand_ratio
                ratio = rand_ratio

        lq_im_patch = torch.clip(lq_im_patch * ratio, self.zero_clip, 1)
        gt_im_patch = torch.clip(gt_im_patch, 0, 1)

        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': torch.tensor(ratio).float(),
            'iso': torch.tensor(iso).float(),
            'wb': gt_wb if self.which_meta == 'gt' else lq_wb,
            'ccm': gt_ccm if self.which_meta == 'gt' else lq_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)