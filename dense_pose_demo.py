
import os
import glob
import cv2
from one_click_densepose.utils.helper import Predictor
from one_click_densepose.un_unwrapping import get_texture
####################################################
# Usage Guide (Jan. 16, 2024)
# ------------------------------------------------
# Features:
# > Can render depth maps, color images, normal maps from .obj files
# > .obj files can be scan model or smpl model (option: rendering_mode)
# > Can change renderer (option: renderer = 'trimesh', 'nr', 'opengl', etc)
# > To run the code, set dataset, paths to source(obj) and result directories
#####################################################

if __name__=='__main__':
    data_root = '/home/mpark/data/IOYS_Famoz/DATASET_2024/TH2.1/DIFFUSION'
    path2image = os.path.join(data_root, 'COLOR/DIFFUSE')
    path2save = os.path.join(data_root, 'COLOR/DENSE_UV')

    path2densepose = '/home/mpark/code/DeepScanBooth/one_click_densepose'
    pose_predictor = Predictor(path2densepose)
    skip_exist = True

    data_list = sorted(glob.glob(os.path.join(path2image, '*')))
    for data in data_list:
        print('processing: ' + data)
        img_list = sorted(glob.glob(os.path.join(data, '*.png')))
        dir_name = data.split('/')[-1]
        for image in img_list:
            # densepose-based uv unwrapping
            filename = image.split('/')[-1]
            path2dense_uv = os.path.join(path2save, dir_name, filename)
            if os.path.exists(path2dense_uv) and skip_exist:
                continue

            print('processing... data: %s' % image)
            frame = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            os.makedirs(os.path.join(path2save, dir_name), exist_ok=True)
            iuv, bbox = pose_predictor.predict_for_unwrapping(frame[:, :, :3])
            iuv = iuv.detach().cpu().numpy().transpose(1, 2, 0)
            bbox = bbox.detach().cpu().numpy()[0]
            uv_smpl = get_texture(frame[:, :, :3], iuv[:, :, ::-1], bbox)
            cv2.imwrite(path2dense_uv, uv_smpl)
