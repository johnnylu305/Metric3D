import torch
import time
import os
import numpy as np
import cv2
from PIL import Image
from mono.utils.do_test import transform_test_data_scalecano


def postprocess_per_image(pred_depth, rgb_origin, pad, normalize_scale, scale_info):

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear').squeeze() # to original size
    pred_depth = pred_depth * normalize_scale / scale_info
    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    
    return pred_depth


def get_image(img_path):
    rgb = cv2.imread(img_path)[:, :, ::-1]
    return np.array(rgb)


def get_gt_depth(img_path):
    depth = Image.open(img_path) 
    # mm to m
    depth = np.array(depth)/1000.
    # pgm format is up-side-down
    return depth[::-1]


def compute_rmse(depth_gt, depth_map):
    # Ensure the depth maps have the same shape
    if depth_gt.shape != depth_map.shape:
        print(depth_gt.shape, depth_map.shape)
        raise ValueError("Depth maps must have the same dimensions.")
    
    # mask out unknown depth
    mask = depth_gt != 0 

    # Compute the difference
    diff = depth_gt - depth_map

    # Square the differences
    squared_diff = np.square(diff)

    # Compute the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff[mask])

    # Compute the RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_diff)

    return rmse


class DotDict(dict):
    """Allows dot notation access to dictionary attributes."""
    def __getattr__(self, attr):
        return self.get(attr)
    
    def __setattr__(self, attr, value):
        self[attr] = value
    
    def __delattr__(self, attr):
        del self[attr]


def dict_to_dotdict(d):
    """Recursively converts a dictionary to a DotDict."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_dotdict(DotDict(value))
    return DotDict(d)


def run_metric3d(img_path, model, intrinsic=None, data_basic=None):

    # get rgb image
    rgb_origin = get_image(img_path)

    # default intrinsic for scale factor calculation 
    if intrinsic is None:
        intrinsic = [1000.0, 1000.0, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]
        #intrinsic = [1058.0, 1058.0, 966, 540]

    # default data value
    if data_basic is None:
        data_basic = dict(
        canonical_space = dict(
            # img_size=(540, 960),
            focal_length=1058, #1000.0,
        ),
        depth_range=(0, 1),
        depth_normalize=(0.1, 200),
        crop_size = (616, 1064),  # %28 = 0
        clip_depth_range=(0.1, 200),
        vit_size=(616,1064)
    )

    data_basic = dict_to_dotdict(data_basic)

    # preprocess data including resize and crop
    rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, data_basic)
    # batch dimension
    rgb_inputs = torch.tensor(rgb_input).unsqueeze(0)
    pads = [pad]
    label_scale_factors = [label_scale_factor]

    # run model
    model = torch.hub.load('yvanyin/metric3d', model, pretrain=True, trust_repo=True)
    model = model.cuda().eval()
    with torch.no_grad():
        pred_depths, confidence, outputs = model.inference({'input': rgb_inputs})

    normalize_scale = data_basic.depth_range[1]

    pred_depth = postprocess_per_image(
        pred_depths[0, :],
        rgb_origin,
        pads[0],
        normalize_scale,
        label_scale_factors[0],
    )

    return pred_depth.cpu().numpy()

def main():
    
    img_path = os.path.join("dataset", "image_left.png")
    
    depth_path = os.path.join("dataset", "Depth.pfm")
    depth_gt = get_gt_depth(depth_path)

    
    start = time.time()
    depth_small = run_metric3d(img_path, model="metric3d_vit_small")
    rmse_small = compute_rmse(depth_gt, depth_small)
    rmse_small_crop = compute_rmse(depth_gt[400:], depth_small[400:])
    end_small = time.time() - start
    

    
    start = time.time()
    depth_large = run_metric3d(img_path, model="metric3d_vit_large")
    rmse_large = compute_rmse(depth_gt, depth_large)
    rmse_large_crop = compute_rmse(depth_gt[400:], depth_large[400:])
    end_large = time.time() - start

    start = time.time()
    depth_giant = run_metric3d(img_path, model="metric3d_vit_giant2")
    rmse_giant = compute_rmse(depth_gt, depth_giant)
    rmse_giant_crop = compute_rmse(depth_gt[400:], depth_giant[400:])
    end_giant = time.time() - start

    print(f"Small model TIme {end_small} RMSE {rmse_small} CROP {rmse_small_crop}")

    print(f"Large model TIme {end_large} RMSE {rmse_large} CROP {rmse_large_crop}")

    print(f"GIant model TIme {end_giant} RMSE {rmse_giant} CROP {rmse_giant_crop}")
    
if __name__=="__main__":
    main()
