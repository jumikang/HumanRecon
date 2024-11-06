
import os
import sys
import math
import json
import torch
import pickle
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
                               return_disp=params['return_disp']
                               # validation=validation
                               )

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False if validation else True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True if opt['num_workers'] == 0 else False,
        batch_size=opt['batch_size'],
        num_workers=opt['num_workers'],
    )

class HumanScanDataset(Dataset):
    def __init__(self,
                 root_dir=None,
                 datalist=None,
                 res=512,
                 return_disp=False
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.return_disp = return_disp
        self.paths = []
        if root_dir is not None and datalist is not None:
            list_filename = os.path.join(root_dir, datalist)
            if os.path.exists(list_filename):
                with open(os.path.join(root_dir, datalist)) as f:
                    self.paths = json.load(f)

        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = transforms.Compose(
            [
                transforms.Resize(res),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # (0,1)->(-1,1)
            ]
        )

    @staticmethod
    def load_im(path, bg_color=None):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()

        if img.shape[2] == 4 and bg_color is not None:
            img[img[:, :, -1] == 0.] = bg_color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

    def __len__(self):
        return len(self.paths)

    def load_in_the_wild(self, img, dense_uv):
        cond_posed = self.process_im(Image.fromarray(img))
        cond_dense = self.process_im(Image.fromarray(dense_uv))
        return {"image_cond":cond_posed[None, ...],
                "dense_cond": cond_dense[None, ...]}

    def __getitem__(self, index):
        data = {}
        path2img_posed = os.path.join(self.root_dir, self.paths[index])
        path2img_dense = path2img_posed.replace('DIFFUSE', 'DENSE_UV')

        f_name = path2img_posed.split('/')[-1]
        # dir_name, f_name = filename.split('/')[-2:]

        # color = [1., 1., 1., 1.]
        bg_color = None
        cond_posed = self.process_im(self.load_im(os.path.join(path2img_posed), bg_color=bg_color))
        cond_dense = self.process_im(self.load_im(path2img_dense, bg_color=bg_color))

        label_img = path2img_posed.replace('DIFFUSION/COLOR/DIFFUSE/',
                                           '/UV_CANON/').replace(f_name, 'material_0.png')
        target_im = self.process_im(self.load_im(label_img, bg_color=bg_color))

        data["image_target"] = target_im
        data["image_cond"] = cond_posed
        data["dense_cond"] = cond_dense

        if self.return_disp:
            disp = np.load(label_img.replace('material_0.png', 'disp.pkl'), allow_pickle=True)
            data["disp_target"] = torch.FloatTensor(disp).transpose(1, 0)

        return data
