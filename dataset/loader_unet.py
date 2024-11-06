import os
import sys
import math
import json
import torch
import trimesh
import pickle
import cv2
import yaml
import smplx
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

def create_dataset(params, validation=False):
    """
    Create HumanScanDataset from config.yaml
    :param params: train and validation parameters
    :param validation: create different datasets for different configurations
    :return: dataloader instance
    """
    opt = params['validation'] if validation else params['train']
    dataset = HumanScanDataset(root_dir=params['root_dir'],
                               datalist=opt['data_list'],
                               res=params['res'],
                               v_interval=params['v_interval'],
                               h_interval=params['h_interval'],
                               pred_canon=params['pred_canon'],
                               dr_loss=params['dr_loss'],
                               return_uv=params['return_uv'],
                               return_disp=params['return_disp'],
                               validation=validation)

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False if validation else True,
        drop_last=True,
        pin_memory=False,
        persistent_workers=True if opt['num_workers'] == 0 else False,
        batch_size=opt['batch_size'],
        num_workers=opt['num_workers'],
    )

class HumanScanDataset(Dataset):
    def __init__(self,
                 root_dir='.objaverse/hf-objaverse-v1/views',
                 datalist= 'list/ZERO123_TH2.0.json',
                 res=512,
                 v_interval=[0],
                 h_interval=[0],
                 pred_canon=False,
                 dr_loss=False,
                 return_disp=False,
                 return_uv=False,
                 validation=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.return_disp = return_disp
        self.return_uv = return_uv
        with open(os.path.join(root_dir, datalist)) as f:
            self.paths = json.load(f)
        self.res = res
        self.v_interval = v_interval
        self.h_interval = h_interval
        self.pred_canon = pred_canon
        self.dr_loss = dr_loss
        self.RGB_MAX = np.array([255.0, 255.0, 255.0])
        self.RGB_MEAN = np.array([0.485, 0.456, 0.406])  # vgg mean
        self.RGB_STD = np.array([0.229, 0.224, 0.225])  # vgg std

        # total_objects = len(self.paths)
        #if validation:
        #    self.paths = self.paths[math.floor(total_objects / 100. * 99.):]  # used last 1% as validation
        #else:
        #    self.paths = self.paths[:math.floor(total_objects / 100. * 99.)]  # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))

    def load_image(self, path, wa=False):
        # return: numpy array
        image = np.array(Image.open(path))

        if not image.shape[1] == self.res:
            image = cv2.resize(image, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        if wa:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_normal(self, path, wa=False):
        normal = np.array(Image.open(path))
        if not normal.shape[1] == self.res:
            normal = cv2.resize(normal, dsize=(self.res, self.res), interpolation=cv2.INTER_AREA)
        if wa:
            normal = cv2.cvtColor(normal, cv2.COLOR_BGRA2RGB)
        else:
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        return normal

    def load_dr(self, im, cam_nums=2):
        dr_img_path = (im.replace('DIFFUSE', 'ALBEDO').replace('_front.', '.'))
        dr_normal_path = dr_img_path.replace('COLOR', 'NORMAL')
        dr_depth_path = dr_img_path.replace('COLOR', 'DEPTH')

        dr_img_paths = []
        dr_normal_paths = []
        dr_depth_paths = []
        h = torch.zeros(cam_nums)
        v = torch.zeros(cam_nums)
        for i in range(0, cam_nums):
            h_degree = random.randrange(0, 360, 2)
            v_degree = random.randrange(-10, 11, 10)
            h[i] = h_degree
            v[i] = v_degree

            if v_degree<0:
                v_degree += 360

            file_name = dr_img_path.split('/')[-1]
            rot_file_name = '%03d_%03d_000.png' % (v_degree, h_degree)
            dr_img = dr_img_path.replace(file_name, rot_file_name)
            dr_normal = dr_normal_path.replace(file_name, rot_file_name)
            dr_depth = dr_depth_path.replace(file_name, rot_file_name)
            dr_img_paths.append(dr_img)
            dr_normal_paths.append(dr_normal)
            dr_depth_paths.append(dr_depth)

        return dr_img_paths, dr_normal_paths, dr_depth_paths, h, v

    def process_im(self, im, wa=False):
        # return: torch tensor
        image = self.load_image(im, wa=wa)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        # image_tensor = torch.flipud(torch.FloatTensor(image)).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))
        return scaled_image_tensor

    def process_dr_im(self, im, wa=False):
        # return: torch tensor
        image = self.load_image(im, wa=wa)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        image_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        scaled_image_tensor =\
            ((image_tensor - torch.FloatTensor(self.RGB_MEAN).view(3, 1, 1))
             / torch.FloatTensor(self.RGB_STD).view(3, 1, 1))
        return scaled_image_tensor

    def process_dr_normal(self, normal, wa=False):
        # return: torch tensor
        normal = self.load_normal(normal, wa=wa)
        normal_tensor = torch.FloatTensor(normal).permute(2, 0, 1)
        normal_tensor /= torch.FloatTensor(self.RGB_MAX).view(3, 1, 1)
        normal_tensor = normal_tensor * 2 - 1
        return normal_tensor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = {}

        # input data load
        path2img_posed = os.path.join(self.root_dir, self.paths[index])
        path2img_posed = path2img_posed.replace('/./', '/')
        path2img_dense = path2img_posed.replace('DIFFUSE', 'PRED_UV')

        cond_posed = self.process_im(path2img_posed, wa=True)
        cond_dense = self.process_im(path2img_dense, wa=False)
        data["image_cond"] = cond_posed
        data["uv_cond"] = cond_dense

        # target data load
        f_name = path2img_posed.split('/')[-1]
        data_name = path2img_posed.split('/')[-2]

        label_img = (path2img_posed.replace('CES/COLOR/DIFFUSE/', 'UV_CANON/')
                     .replace(f_name, 'material_0.png'))
        mesh_dic = np.load(label_img.replace('material_0.png', 'meshes_info.pkl'), allow_pickle=True)
        mesh_param_path = (label_img.replace('UV_CANON', 'SMPLX_UV')
                           .replace('material_0.png', '%s.json' % data_name))
        if self.return_uv:
            target_im = self.process_im(label_img, wa=False)
            data["uv_target"] = target_im

        if self.return_disp:
            if self.pred_canon:
                data["disp_target"] = torch.FloatTensor(mesh_dic['canon_disp']).transpose(1, 0)
            else:
                data["disp_target"] = torch.FloatTensor(mesh_dic['pose_disp']).transpose(1, 0)

        if self.dr_loss:
            data["smpl_param_path"] = mesh_param_path
            dr_img_paths, dr_normal_paths, dr_depth_paths, angle_h, angle_v \
                = self.load_dr(path2img_posed)
            dr_imgs = []
            dr_normals = []
            for i in range(len(dr_img_paths)):
                dr_imgs.append(self.process_dr_im(dr_img_paths[i], wa=True))
                dr_normals.append(self.process_dr_normal(dr_normal_paths[i], wa=True))
            data["dr_imgs_target"] = dr_imgs
            data["dr_normals_target"] = dr_normals
            data["angle_h"] = torch.FloatTensor(angle_h)
            data["angle_v"]= torch.FloatTensor(angle_v)

            if self.pred_canon:
                data["disp_target"] = torch.FloatTensor(mesh_dic['canon_disp']).transpose(1, 0)
                data["smpl_vertices"] = torch.FloatTensor(mesh_dic['canon_verts'])
                data["smpl_faces"] = torch.Tensor(mesh_dic['faces'])
                data["smpl_uv_vts"] = torch.FloatTensor(mesh_dic['uv_vts'])
                data["lbs_weights"] = torch.FloatTensor(mesh_dic['lbs_weights'])
                data["A"] = torch.FloatTensor(mesh_dic['A'])
            else:
                data["disp_target"] = torch.FloatTensor(mesh_dic['pose_disp']).transpose(1, 0)
                data["smpl_vertices"] = mesh_dic['posed_verts']
                data["smpl_faces"] = mesh_dic['faces']
                data["smpl_uv_vts"] = mesh_dic['uv_vts']

        return data