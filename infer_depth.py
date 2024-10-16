import torch
import time
import os
from PIL import Image


def get_metric3d_depth(rgb, model="metric3d_vit_small"):
    model = torch.hub.load('yvanyin/metric3d', model, pretrain=True, trust_repo=True)
    pred_depth, confidence, output_dict = model.inference({'input': rgb})
    return pred_depth


def get_image(img_path):
    rgb = Image.open(img_path) 
    return rgb


def get_gt_depth(img_path):
    with open(file, 'rb') as f:
        # Read header
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Read dimensions
        dim_line = f.readline().decode('utf-8')
        width, height = map(int, re.findall(r'\d+', dim_line))

        # Read scale (also determines endianness)
        scale_line = f.readline().decode('utf-8').rstrip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'  # little endian if scale is negative

        # Read pixel data
        data = np.fromfile(f, endian + 'f')

        # Reshape the data
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM stores pixels starting from bottom-left corner
    # not sure why the depth is up-side-down
    return data[::-1]


def compute_rmse(depth_gt, depth_map):
    # Ensure the depth maps have the same shape
    if depth_gt.shape != depth_map.shape:
        raise ValueError("Depth maps must have the same dimensions.")

    # mask out unknown depth
    mask = depth_gt != 0 

    # Compute the difference
    diff = depth_map1 - depth_map2

    # Square the differences
    squared_diff = np.square(diff)

    # Compute the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff[mask])

    # Compute the RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_diff)

    return rmse


def main():
    
    rgb_path = os.path.join("dataset", "image_left.png")
    rgb = get_image(rgb_path)

    depth_path = os.path.join("dataset", "Depth.pfm")
    depth_gt = get_image(depth_path)

    start = time.time()
    depth_small = get_metric3d_depth(rgb, model="metric3d_vit_small")
    rmse_small = compute_rmse(depth_gt, depth_small)
    end_small = time.time() - start

    start = time.time()
    depth_large = get_metric3d_depth(rgb, model="metric3d_vit_large")
    rmse_large = compute_rmse(depth_gt, depth_small)
    end_large = time.time() - start

    start = time.time()
    depth_giant = get_metric3d_depth(rgb, model="metric3d_vit_giant")
    rmse_giant = compute_rmse(depth_gt, depth_small)
    end_giant = time.time() - start

    print(f"Small model TIme {end_small} RMSE {rmse_small}")

    print(f"Large model TIme {end_large} RMSE {rmse_large}")

    print(f"GIant model TIme {end_giant} RMSE {rmse_giant}")

if __name__=="__main__":
    main()
