import time
import cv2
import os
import glob
import pickle
import matplotlib
import numpy as np
import subprocess
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from one_click_densepose.utils.helper import Predictor
from UVTextureConverter import Atlas2Normal

class DensePoser:
    def __init__(self, path2densepose):
        self.pose_predictor = Predictor(path2densepose)

    def get_dense_uv(self, frame):
        iuv, bbox = self.pose_predictor.predict_for_unwrapping(frame[:, :, :3])
        iuv = iuv.detach().cpu().numpy().transpose(1, 2, 0)
        bbox = bbox.detach().cpu().numpy()[0]
        uv_smpl = get_texture(frame[:, :, :3], iuv[:, :, ::-1], bbox)
        return uv_smpl

def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv

def parse_bbox(result):
    return result["pred_boxes_XYXY"][0].cpu().numpy()

def concat_textures(array):
    texture = []
    for i in range(4):
        tmp = array[6 * i]
        for j in range(6 * i + 1, 6 * i + 6):
            tmp = np.concatenate((tmp, array[j]), axis=1)
        texture = tmp if len(texture) == 0 else np.concatenate((texture, tmp), axis=0)
    return texture

def interpolate_tex(tex):
    # code is adopted from https://github.com/facebookresearch/DensePose/issues/68
    valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
    radius_increase = 10
    kernel = np.ones((radius_increase, radius_increase), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    region_to_fill = dilated_mask - valid_mask
    invalid_region = 1 - valid_mask
    actual_part_max = tex.max()
    actual_part_min = tex.min()
    actual_part_uint = np.array((tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                                   cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
    # only use dilated part
    actual_part = actual_part * dilated_mask

    return actual_part

def get_texture(im, iuv, bbox, tex_part_size=200):
    # this part of code creates iuv image which corresponds
    # to the size of original image (iuv from densepose is placed
    # within pose bounding box).
    im = im.transpose(2, 1, 0) / 255
    image_w, image_h = im.shape[1], im.shape[2]
    x, y, w, h = [int(v) for v in bbox]
    bg = np.ones((image_h, image_w, 3))
    bg[y:y + h, x:x + w, :] = iuv
    iuv = bg
    iuv = iuv.transpose((2, 1, 0))
    i, u, v = iuv[2], iuv[1], iuv[0]

    # following part of code iterate over parts and creates textures
    # of size `tex_part_size x tex_part_size`
    n_parts = 24
    texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))

    for part_id in range(1, n_parts + 1):
        generated = np.zeros((3, tex_part_size, tex_part_size))

        x, y = u[i == part_id], v[i == part_id]
        # transform uv coodrinates to current UV texture coordinates:
        tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
        tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)

        # clipping due to issues encountered in denspose output;
        # for unknown reason, some `uv` coos are out of bound [0, 1]
        tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
        tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)

        # write corresponding pixels from original image to UV texture
        # iterate in range(3) due to 3 chanels
        for channel in range(3):
            generated[channel][tex_v_coo, tex_u_coo] = im[channel][i == part_id]

        # this part is not crucial, but gives you better results
        # (texture comes out more smooth)
        if np.sum(generated) > 0:
            generated = interpolate_tex(generated)

        # assign part to final texture carrier
        texture[part_id - 1] = generated[:, ::-1, :]

    # concatenate textures and create 2D plane (UV)
    tex_concat = np.zeros((24, tex_part_size, tex_part_size, 3))

    for i in range(texture.shape[0]):
        tex_concat[i] = texture[i].transpose(2, 1, 0)
    # tex = concat_textures(tex_concat)

    converter = Atlas2Normal(atlas_size=200, normal_size=1024)
    normal_tex = converter.convert((tex_concat * 255).astype('int'), mask=None)
    return np.uint8(normal_tex * 255)

if __name__=='__main__':
    flag = False # True: pickle extraction, False: smpl-based uv image extraction
    data_path = '/home/mpark/data/IOYS_Famoz/DATASET_2024'
    dataset = 'TH2.1'
    input_path = os.path.join(data_path, dataset, 'DIFFUSION', 'COLOR', 'DIFFUSE')
    intermediate_path = os.path.join(data_path, dataset, 'DIFFUSION', 'COLOR', 'DENSEPOSE')
    out_path = os.path.join(data_path, dataset, 'DENSE_UV')

    data_list = sorted(os.listdir(input_path))
    skip_exist = True
    if flag:
        for i in data_list:
            img_list = sorted(glob.glob(os.path.join(input_path, i, '*.png')))
            for img_data in img_list:
                pkl_path = img_data.replace('.png', '.pkl').replace('DIFFUSE', 'DENSEPOSE')
                if skip_exist and os.path.exists(pkl_path):
                    continue

                os.makedirs(os.path.join(intermediate_path, i), exist_ok=True)
                cmd = 'python3 apply_net.py dump configs/densepose_rcnn_R_101_FPN_s1x.yaml models/R_101_FPN_s1x.pkl %s --output output -v' % img_data
                pkl_data = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                print('processing... data: %s' % img_data.split('/')[-2:])
                time.sleep(4)
                pkl_data.terminate()
    else:
        for i in data_list:
            img_list = sorted(glob.glob(os.path.join(input_path, i, '*.png')))
            for img_data in img_list:
                # img_data = os.path.join(input_path, i, '0_0_00.png')
                pkl_path = pkl_path = img_data.replace('.png', '.pkl').replace('DIFFUSE', 'DENSEPOSE')
                with open(pkl_path, "rb") as fr:
                    data = pickle.load(fr)
                results = data[0]

                iuv = parse_iuv(results)
                bbox = parse_bbox(results)
                image = cv2.imread(img_data)[:, :, ::-1]
                uv_texture, uv_smpl = get_texture(image, iuv, bbox)
                uv_img = Image.fromarray(np.uint8(uv_texture*255), "RGB")

                save_path = os.path.join(out_path, i)
                os.makedirs(os.path.join(out_path, i), exist_ok=True)
                uv_smpl.save(os.path.join(save_path, '0_0_00.png'))
                print('processing... data: %s' % i)