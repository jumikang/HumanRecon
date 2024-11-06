import os
import json
import glob
import random
# import cv2
import numpy as np
# from depth_anything_v2.dpt import DepthAnythingV2

def split_list_json(data_root=None, filename=None, mode='TRAIN'):
    with open(os.path.join(data_root, '%s.json' % filename), "r") as f:
        all_files = json.load(f)

        # split TH2.1(and TH2.0) dataset
        train_files = all_files[:35000] + all_files[36820:]
        val_files = all_files[35000:36820]  # exclude TH500~525 (26 models)
        val_files = val_files[::35]  # two lighting for each object.

        with open(os.path.join(data_root, filename.replace(filename, '%s_TRAIN' % mode)) + '.json', "w") as f_train:
            json.dump(train_files, f_train)
        with open(os.path.join(data_root, filename.replace(filename, '%s_VAL' % mode)) + '.json', "w") as f_val:
            json.dump(val_files, f_val)

def generate_list_diffusion(data_root='/home/somewhere/in/your/pc', filename=None, datalist=None, mode='TRAIN'):
    path2list = os.path.join(data_root, 'list')
    if os.path.isdir(path2list) is False and os.path.isfile(path2list) is False:
        os.mkdir(path2list)

    all_files = os.path.join(path2list, filename) + ".json"
    folder_list = []
    for dataname in datalist:
        folder_list += sorted(glob.glob(os.path.join(data_root, dataname, mode, 'COLOR/DIFFUSE/*')))

    img_list = []
    for folder in folder_list:
        img_list += sorted(glob.glob(os.path.join(folder, '*.png')))

    img_list = [img.replace(data_root, '.') for img in img_list]
    with open(all_files, "w") as f:
        json.dump(img_list, f)

# generate training related files when this function is called as main.
if __name__ == '__main__':
    data_path = '/home/mpark/data/IOYS_Famoz/DATASET_2024'
    # data_name = ['RP', 'TH2.0', 'TH3.0', 'IOYS_T', 'IOYS_4090']
    data_name = ['TH2.1', 'RP', 'IOYS_T']
    # data_name = ['RP', 'TH2.1', 'IOYS_T', 'IOYS_4090', 'IOYS_7500_0_3',
    #              'IOYS_7500_1_3', 'IOYS_7500_2_3', 'IOYS_7500_3_3']
    mode = 'DIFFUSION'  # 'TRAIN', 'VAE'
    filename = 'DIFFUSION'
    for data in data_name:
        generate_list_diffusion(data_root=data_path, filename=filename +
                              '_' + data, datalist=[data], mode=mode)
    generate_list_diffusion(data_root=data_path, filename=filename +
                          '_all', datalist=data_name, mode=mode)
    split_list_json(data_root=os.path.join(data_path, 'list'), filename=filename+'_all', mode=mode)
    # dataset_list = ['RP', 'TH2.0', 'TH3.0']
    # merge_train_files(os.path.join(data_path, 'list'), dataset_list, mode=mode, prefix='MERGED')
    # generate_test_list_files(data_root=data_path, filename=filename, dataname=data_name)
