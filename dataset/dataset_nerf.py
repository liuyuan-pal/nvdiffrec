# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json
from pathlib import Path

import torch
import numpy as np
from skimage.io import imread, imsave

from dataset.base_utils import read_pickle, save_pickle
from colmap.read_write_model import read_model
from render import util

from .dataset import Dataset

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################
from .gen6d_utils import Gen6D_ROOT, Gen6DMetaInfoWrapper, get_projected_mask


def _load_img(path):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERF(Dataset):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetNERF: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img      = []
        fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv, # model view
            'mvp' : mvp, # model view projection
            'campos' : campos, # camera position
            'resolution' : iter_res, # resolution
            'spp' : self.FLAGS.spp, # seems useless = 1
            'img' : img # note this image is transformed by srgb_to_rgb
        }


class DatasetGen6D(Dataset):
    def __init__(self, seq_name, FLAGS, is_train):
        super().__init__()
        self.FLAGS = FLAGS
        self.seq_name = seq_name
        self.root = Path(Gen6D_ROOT) / (self.seq_name + '-ref')
        self.image_path = (Path(Gen6D_ROOT) / (seq_name + '-ref') / 'images')
        img_fns_cache = self.root / 'images_fn_cache.pkl'
        if img_fns_cache.exists():
            self.img_fns = read_pickle(str(img_fns_cache))
        else:
            self.img_fns = [fn for fn in os.listdir(str(self.image_path)) if fn.endswith('.jpg')]
            save_pickle(self.img_fns, str(img_fns_cache))

        self.img_ids = [str(k) for k in range(len(self.img_fns))]
        self.meta_info = Gen6DMetaInfoWrapper(seq_name)

        self._load_colmap_data()
        self._get_opengl_mvp()
        self._compute_mask()

        val_ids = self.img_ids[::20]
        train_ids = [img_id for img_id in self.img_ids if img_id not in val_ids]
        if is_train:
            self.img_ids = train_ids
        else:
            self.img_ids = val_ids
        self.is_train = is_train

    def _get_opengl_mvp(self):
        self.mv, self.p = {}, {}
        h_ = np.asarray([[0,0,0,1]],np.float32)
        self.res=0
        self.Ks={}
        for img_id, pose in self.poses.items():
            # normalize poses so that object is in the unit sphere
            self.poses[img_id] = self.meta_info.normalize_pose(pose)

            f, cx, cy = self.intrinsics[img_id]
            K = np.asarray([[f,0,cx],[0,f,cy],[0,0,1]],np.float32)
            self.Ks[img_id] = K

            # transform to modelview matrix
            # mv = np.diag([1,-1,-1]).astype(np.float32) @ self.poses[img_id]
            mv = np.diag([1,-1,-1]).astype(np.float32) @ self.poses[img_id]
            mv = np.concatenate([mv,h_], 0)
            self.mv[img_id] = mv

            # projection matrix
            focal, cx, cy = self.intrinsics[img_id]
            h, w, _ = imread(str(self.image_path/self.img_fns[int(img_id)])).shape

            n=0.1
            f=1000.0
            # l=-cx/focal*n
            # r=(w-cx)/focal*n
            # b=-cy/focal*n
            # t=(h-cy)/focal*n

            # 2n/(r-l)=2n/(w-cx+cx)*focal/n=2focal/w
            # 2n
            # r+l=((w-2cx))/(w)

            project_mat = np.asarray([
                [2*focal/w,  0,   (w-2*cx)/w,            0],
                [0, -2*focal/h,   (h-2*cy)/h,            0],
                [0,          0, -(f+n)/(f-n), -2*f*n/(f-n)],
                [0,          0,           -1,            0]
                ], dtype=np.float32
            )
            self.p[img_id] = project_mat
            # do we need to scale the depth range to real depth range?
            self.res = [h, w]

    def _compute_mask(self):
        self.mask_dir=f'{str(self.root)}/masks_proj'
        Path(self.mask_dir).mkdir(exist_ok=True,parents=True)
        # compute mask from projected point clouds
        for img_id, pose in self.poses.items():
            if (Path(self.mask_dir) / f'{img_id}.png').exists():
                continue
            K = self.Ks[img_id]
            h, w, _ = imread(str(self.image_path/self.img_fns[int(img_id)])).shape
            mask = get_projected_mask(self.meta_info.object_point_cloud, self.poses[img_id], K, h, w)
            mask = np.asarray(mask,np.uint8)
            imsave(f'{self.mask_dir}/{img_id}.png', mask)

    def _load_colmap_data(self):
        # load colmap model
        cameras, images, points3d = read_model(f'{Gen6D_ROOT}/{self.seq_name}-ref/colmap-all/colmap_default-colmap_default/sparse/0')
        img_id2db_id = {v.name[:-4]:k for k, v in images.items()}
        self.img_ids = [img_id for img_id in self.img_ids if img_id in img_id2db_id]

        # poses
        self.poses, self.intrinsics = {}, {}
        for img_id in self.img_ids:
            db_id = img_id2db_id[img_id]
            R = images[db_id].qvec2rotmat()
            t = images[db_id].tvec
            pose = np.concatenate([R,t[:,None]],1).astype(np.float32) # world to opencv camera
            self.poses[img_id] = pose

            # project matrix
            cam_id = images[db_id].camera_id
            f, cx, cy, _ = cameras[cam_id].params
            self.intrinsics[img_id] = [f,cx,cy]

    def __len__(self):
        if self.is_train:
            return (self.FLAGS.iter+1) * self.FLAGS.batch
        else:
            return len(self.img_ids)

    def __getitem__(self, index):
        index %= len(self.img_ids)
        mv = torch.from_numpy(self.mv[self.img_ids[index]]).float()
        mvp = torch.from_numpy(self.p[self.img_ids[index]]).float() @ mv
        campos = torch.inverse(mv)[:3,3]
        img = imread(str(self.image_path/self.img_fns[int(self.img_ids[index])]))
        img = torch.tensor(img.astype(np.float32) / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])

        # read mask as alpha
        mask = imread(f'{self.mask_dir}/{self.img_ids[index]}.png')
        mask = mask.astype(np.float32) /255
        mask = torch.from_numpy(mask)
        img = torch.cat([img, mask[..., None]], -1)

        return {
            'mv': mv[None],  # model view
            'mvp': mvp[None],  # model view projection
            'campos': campos[None],  # camera position
            'resolution': self.res,  # resolution
            'spp': self.FLAGS.spp,  # seems useless = 1
            'img': img[None]  # note this image is transformed by srgb_to_rgb
        }