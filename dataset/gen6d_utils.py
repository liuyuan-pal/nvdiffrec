import abc
import os
from pathlib import Path

import cv2
import numpy as np
import plyfile
import xml.etree.cElementTree as ET

from skimage.io import imread, imsave
from tqdm import tqdm

from colmap.read_write_model import read_model
from dataset.base_utils import read_pickle, triangulate, save_pickle, project_points, pose_inverse
from dataset.pose_utils import let_me_look_at, look_at_crop

Gen6D_ROOT= './data/gen6d'

def load_point_cloud(pcl_path):
    with open(pcl_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
    return xyz

gen6d_meta_info={
    'cup': {'gravity': np.asarray([-0.0893124,-0.399691,-0.912288]), 'forward': np.asarray([-0.009871,0.693020,-0.308549],np.float32)},
    'warrior': {'gravity': np.asarray([-0.0734401,-0.633415,-0.77032]), 'forward': np.asarray([-0.121561, -0.249061, 0.211048],np.float32)},
    'chair': {'gravity': np.asarray((0.111445, -0.373825, -0.920779),np.float32), 'forward': np.asarray([0.788313,-0.139603,0.156288],np.float32)},
    'knife': {'gravity': np.asarray((-0.0768299, -0.257446, -0.963234),np.float32), 'forward': np.asarray([0.954157,0.401808,-0.285027],np.float32)},
    'love': {'gravity': np.asarray((0.131457, -0.328559, -0.93529),np.float32), 'forward': np.asarray([-0.045739,-1.437427,0.497225],np.float32)},
    'plug_cn': {'gravity': np.asarray((-0.0267497, -0.406514, -0.913253),np.float32), 'forward': np.asarray([-0.172773,-0.441210,0.216283],np.float32)},
    'plug_en': {'gravity': np.asarray((0.0668682, -0.296538, -0.952677),np.float32), 'forward': np.asarray([0.229183,-0.923874,0.296636],np.float32)},
    'rabbit': {'gravity': np.asarray((-0.153506, -0.35346, -0.922769),np.float32), 'forward': np.asarray([-0.584448,-1.111544,0.490026],np.float32)},
    'razor': {'gravity': np.asarray((-0.122099,-0.496839, -0.85921),np.float32), 'forward': np.asarray([-1.520792,0.640192,0.293668],np.float32)},
    'scissors': {'gravity': np.asarray((-0.129767, -0.433414, -0.891803),np.float32), 'forward': np.asarray([1.899760,0.418542,-0.473156],np.float32)},
    'piggy': {'gravity': np.asarray((-0.122392, -0.344009, -0.930955), np.float32), 'forward': np.asarray([0.079012,1.441836,-0.524981], np.float32)},
    'plug_en2': {'gravity': np.asarray((-0.15223,-0.205333,-0.96678), np.float32), 'forward': np.asarray([ 1.258604, 0.206405,-0.247187], np.float32)},
    'chair2': {'gravity': np.asarray((-0.0803511,-0.120156,-0.989498), np.float32), 'forward': np.asarray([-1.074559,0.946239,-0.075782], np.float32)},
}

class Gen6DMetaInfoWrapper:
    def __init__(self, object_name):
        self.object_name = object_name
        self.gravity = gen6d_meta_info[self.object_name]['gravity']
        self.forward = gen6d_meta_info[self.object_name]['forward']
        self.object_point_cloud = load_point_cloud(f'{Gen6D_ROOT}/{self.object_name}-ref/object_point_cloud.ply')

        # rotate
        self.rotation = self.compute_rotation(self.gravity, self.forward)
        self.object_point_cloud = (self.object_point_cloud @ self.rotation.T)

        # scale
        self.scale_ratio = self.compute_normalized_ratio(self.object_point_cloud)
        self.object_point_cloud *= self.scale_ratio

        # offset
        min_pt = np.min(self.object_point_cloud,0)
        max_pt = np.max(self.object_point_cloud,0)
        self.center = (max_pt + min_pt)/2
        self.object_point_cloud -= self.center[None,:]

        test_fn = f'{Gen6D_ROOT}/{self.object_name}-ref/test-object_point_cloud.ply'
        if Path(test_fn).exists(): self.test_object_point_cloud = load_point_cloud(test_fn)

    @staticmethod
    def compute_normalized_ratio(pc):
        min_pt = np.min(pc,0)
        max_pt = np.max(pc,0)
        dist = np.linalg.norm(max_pt - min_pt)
        scale_ratio = 2.0 / dist
        return scale_ratio

    def normalize_pose(self, pose):
        # x_cam = R @ x_wrd + t
        # x_wrd_new = (R_ @ x_wrd) * s - t_
        # x_wrd = R_.T @ (x_wrd_new + t_)/s
        # x_cam = R @ (R_.T @ x_wrd_new + R_.T @ t_) + t * s
        R = pose[:3,:3]
        t = pose[:3,3:]
        t = self.scale_ratio * t + R @ self.rotation.T @ self.center[:,None]
        R = R @ self.rotation.T
        return np.concatenate([R,t], 1).astype(np.float32)

    @staticmethod
    def compute_rotation(vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R

    @staticmethod
    def _parse_fad(fn,):
        tree = ET.ElementTree(file=fn)
        root = tree.getroot()
        features = root[0][0][0]
        keypoints = []
        for feature in features:
            x = int(feature.attrib['x'])
            y = int(feature.attrib['y'])
            keypoints.append((x, y))
        return keypoints

class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id): # gt poses
        pass

    @abc.abstractmethod
    def get_depth_range(self,img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

class Gen6DDatabase(BaseDatabase):
    # this database is only used for colmap
    def __init__(self, database_name):
        super().__init__(database_name)
        _, seq_name = database_name.split('/')
        self.seq_name = seq_name
        self.root = Path(Gen6D_ROOT) / self.seq_name
        image_path = (Path(Gen6D_ROOT) / seq_name / 'images')
        img_fns_cache = self.root / 'images_fn_cache.pkl'
        if img_fns_cache.exists():
            self.img_fns = read_pickle(str(img_fns_cache))
        else:
            self.img_fns = [fn for fn in os.listdir(str(image_path)) if fn.endswith('.jpg')]
            save_pickle(self.img_fns, str(img_fns_cache))

    def get_image(self, img_id, ref_mode=False):
        return imread(str(self.root / 'images' / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        raise NotImplementedError

    def get_pose(self, img_id):
        raise NotImplementedError

    def get_img_ids(self, check_depth_exist=False):
        return [str(k) for k in range(len(self.img_fns))]

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        raise NotImplementedError

    def get_mask(self, img_id):
        raise NotImplementedError

    def get_depth_range(self, img_id):
        raise NotImplementedError

def get_projected_mask(pc,pose,K,h,w):
    pts2d = project_points(pc, pose, K)[0]
    pts2d = np.round(pts2d).astype(np.int32)
    xs, ys = pts2d[:, 0], pts2d[:, 1]
    xs[xs < 0] = 0; xs[xs >= w] = w - 1
    ys[ys < 0] = 0; ys[ys >= h] = h - 1
    mask = np.zeros([h, w], np.uint8)
    mask[ys, xs] = 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

class Gen6DRefDatabase(BaseDatabase):
    # this class is only in charge read out the poses from colmap
    def __init__(self, database_name):
        super().__init__(database_name)
        _, object_name = database_name.split('/') # gen6d_colmap/object_name
        database_origin = Gen6DDatabase(f'gen6d/{object_name}-ref')
        self.root = database_origin.root
        self.img_fns = database_origin.img_fns

        # parse colmap project
        cameras, images, points3d = read_model(f'{Gen6D_ROOT}/{object_name}-ref/colmap-all/colmap_default-colmap_default/sparse/0')
        img_id2db_id = {v.name[:-4]:k for k, v in images.items()}
        self.poses, self.Ks = {}, {}
        self.img_ids = []
        for img_id in database_origin.get_img_ids():
            if img_id not in img_id2db_id:
                continue
            self.img_ids.append(img_id)
            db_id = img_id2db_id[img_id]
            R = images[db_id].qvec2rotmat()
            t = images[db_id].tvec
            pose = np.concatenate([R,t[:,None]],1).astype(np.float32)
            self.poses[img_id]=pose

            cam_id = images[db_id].camera_id
            f, cx, cy, _ = cameras[cam_id].params
            self.Ks[img_id] = np.asarray([ [f,0,cx], [0,f,cy], [0,0,1],],np.float32)

        self.meta_info = Gen6DMetaInfoWrapper(object_name)
        self.ref_points = self.meta_info.object_point_cloud
        self.range_dict = self._cache_depth_range()

    def _cache_depth_range(self):
        range_fn = (Path(self.root)/'object_depth_range.pkl')
        if range_fn.exists():
            return read_pickle(str(range_fn))
        range_dict={}
        for img_id in self.img_ids:
            K, pose = self.get_K(img_id), self.get_pose(img_id)
            _, depth = project_points(self.ref_points, pose, K)
            near = np.min(depth)*0.9
            far = np.max(depth)*1.1
            range_dict[img_id]=np.asarray([near,far],np.float32)

        save_pickle(range_dict, str(range_fn))
        return range_dict

    def get_image(self, img_id, ref_mode=False):
        return imread(str(self.root / 'images' / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.meta_info.normalize_pose(self.poses[img_id].copy())

    def get_img_ids(self, check_depth_exist=False):
        return [img_id for img_id in self.img_ids]

    def get_depth_range(self, img_id):
        return self.range_dict[img_id]

class Gen6DCropDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, object_name, size = database_name.split('/')
        self.size = int(size)
        self.base_database = Gen6DRefDatabase(f'gen6d_ref/{object_name}')
        self.img_path = Path(self.base_database.root) / f'images_{self.size}'
        self.ref_points = self.base_database.ref_points
        self._cache_images()
        self._cache_masks()

    def _cache_images(self):
        self.img_path.mkdir(exist_ok=True,parents=True)
        if (self.img_path/'info.pkl').exists():
            self.img_id2K, self.img_id2pose, self.range_dict = read_pickle(str(self.img_path/'info.pkl'))
        else:
            # prepare this resized image
            self.img_id2K, self.img_id2pose, self.range_dict = {}, {}, {}
            object_center = np.zeros(3,dtype=np.float32)
            object_diameter = 2.0
            margin = 0.05
            for img_id in tqdm(self.base_database.get_img_ids()):
                pose0, K0, img0 = self.base_database.get_pose(img_id), \
                                  self.base_database.get_K(img_id), \
                                  self.base_database.get_image(img_id)
                _, f1 = let_me_look_at(pose0, K0, object_center)
                dist = np.linalg.norm(pose_inverse(pose0)[:, 3] - object_center)
                f0 = self.size * (1 - margin) / object_diameter * dist
                scale = f0 / f1
                position = project_points(object_center[None], pose0, K0)[0][0]
                img1, K1, pose1, pose_rect, H = \
                    look_at_crop(img0, K0, pose0, position, 0, scale, self.size, self.size)
                self.img_id2K[img_id] = K1
                self.img_id2pose[img_id] = pose1

                _, depth = project_points(self.ref_points, pose1, K1)
                near = np.min(depth) * 0.8
                far = np.max(depth) * 1.2
                self.range_dict[img_id] = np.asarray([near, far], np.float32)
                imsave(str(self.img_path/f'{img_id}.jpg'), img1)

            save_pickle((self.img_id2K, self.img_id2pose, self.range_dict),str(self.img_path/'info.pkl'),)

    def _cache_masks(self):
        for img_id in self.get_img_ids():
            h, w, _ = self.get_image(img_id).shape
            mask = get_projected_mask(self.base_database.meta_info.object_point_cloud,
                                      self.get_pose(img_id), self.get_K(img_id), h, w,)
            imsave(str(Path(self.base_database.root) / f'masks_{self.size}' / f'{img_id}.png'), mask)

    def get_image(self, img_id):
        return imread(str(self.img_path/f'{img_id}.jpg'))

    def get_K(self, img_id):
        return self.img_id2K[img_id].copy().astype(np.float32)

    def get_pose(self, img_id):
        return self.img_id2pose[img_id].copy().astype(np.float32)

    def get_depth_range(self, img_id):
        return self.range_dict[img_id]

    def get_img_ids(self):
        return self.base_database.get_img_ids()
