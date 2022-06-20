import nvdiffrast.torch as dr
import collections

import torch
from skimage.io import imsave, imread

import render.renderutils as ru
import numpy as np

from dataset.base_utils import project_points, color_map_backward
from dataset.dataset_nerf import DatasetGen6D
from dataset.draw_utils import draw_keypoints


def check_projection():
    FLAGS = collections.namedtuple("FLAGS", ["spp"])
    dataset = DatasetGen6D('warrior',FLAGS(spp=1),False)
    pts = dataset.meta_info.object_point_cloud
    pts_th = torch.from_numpy(pts.astype(np.float32))
    data = dataset[0]
    mvp = data['mvp']
    img = color_map_backward(data['img'].cpu().numpy())
    pts2d_gl = ru.xfm_points(pts_th[None,:,:].cuda(), mvp.cuda()).cpu().numpy()

    pts2d_gl = pts2d_gl[0,:,:2]/pts2d_gl[0,:,3:]
    pts2d_gl[:,0]*=540/2
    pts2d_gl[:,1]*=960/2
    pts2d_gl[:,0]+=540/2
    pts2d_gl[:,1]+=960/2
    # pts2d_gl[:, 1]=960-pts2d_gl[:,1]
    print(pts2d_gl[:10])

    pose = dataset.poses[dataset.img_ids[0]]
    K = dataset.Ks[dataset.img_ids[0]]
    pts2d, _ = project_points(pts, pose, K)
    print(pts2d[:10])

    np.random.seed(1234)
    idx = np.random.randint(0,pts.shape[0],3)
    pts_ = pts[idx]

    pos = torch.from_numpy(pts_.astype(np.float32)).cuda()
    # h_ = torch.ones([3,1],dtype=torch.float32,device='cuda')
    # pos = torch.cat([pos,h_],1)
    pos = ru.xfm_points(pos[None], mvp.cuda())

    tri = torch.from_numpy(np.asarray([0,1,2],np.int32)).cuda()
    resolution = (960,540)

    glctx = dr.RasterizeGLContext()
    # rast, _ = dr.rasterize(glctx, pos, tri[None], resolution)

    with dr.DepthPeeler(glctx, pos, tri[None], resolution) as peeler:
        rast, db = peeler.rasterize_next_layer()
    mask = rast[0,:,:,3:4]>0
    mask = mask.detach().cpu().numpy().astype(np.uint8)
    mask = mask * img[0]

    pts2d_, _ = project_points(pts_, pose, K)
    imsave('data/vis_val/raw.jpg', draw_keypoints(img[0,...,:3], pts2d_))
    imsave('data/vis_val/tmp.jpg', mask[...,:3])

def generate_detectron_mask():
    import detectron2
    from detectron2 import model_zoo
    from detectron2.projects import point_rend
    from detectron2.engine import DefaultPredictor

    cfg = detectron2.config.get_cfg()
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    predictor = DefaultPredictor(cfg)

    # imread
    img = imread(f'data/gen6d/ms-ref/images/frame0.jpg')
    outputs = predictor(img)
    import ipdb; ipdb.set_trace()

generate_detectron_mask()

# check_projection()