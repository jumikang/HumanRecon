
import cv2
import os
import glob
import hydra
import torch
import numpy as np
import trimesh
import warnings
import pytorch_lightning as pl
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
from omegaconf import OmegaConf
from dataset.loader_diffusion import HumanScanDataset
from one_click_densepose.un_unwrapping import DensePoser
from models.diffusion.base_model import BaseDiffusionModel
from models.unet.deep_human_models import DeepHumanUVNet
from diff_renderer.normal_nds.nds.core.mesh_ext import TexturedMesh


@hydra.main(config_path="config", config_name="base_config_diffusion")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())
    print(OmegaConf.to_yaml(opt))

    # load preprocessing u-net network
    path2ckpt_pre = "/home/mpark/code/uv_recon_ces/outputs/pretrained/last_unet.ckpt"
    check_point_pre = torch.load(path2ckpt_pre)
    model_pre = DeepHumanUVNet(opt)
    model_pre.load_state_dict(check_point_pre['state_dict'])

    # load stable diffusion network
    path2ckpt = "/home/mpark/code/uv_recon_ces/outputs/pretrained/last.ckpt"
    check_point = torch.load(path2ckpt)
    model = BaseDiffusionModel(pretrained_model=opt.model.pretrained_model)
    model.load_state_dict(check_point['state_dict'])
    model.eval()

    # load densepose predictor
    path2densepose = '/home/mpark/code/DeepScanBooth/one_click_densepose'
    densepose_predictor = DensePoser(path2densepose)

    # set dataloader
    dataset = HumanScanDataset()

    is_train = False
    if is_train:
        data_root = "/home/mpark/data/IOYS_Famoz/DATASET_2024/TH2.1/CES/COLOR/DIFFUSE"
        ext = '.png'
    else:
        data_root = "/home/mpark/data2/IOYS_Famoz/STUDIO_SET3/IMG"
        ext = '.jpg'

    datalist = sorted(glob.glob(os.path.join(data_root, '*')))
    data_all = []
    for data in datalist:
        data_all.append(sorted(glob.glob(os.path.join(data, '*'+ext))))

    path2uv_mesh = '/home/mpark/code/uv_recon_ces/resource/smplx_uv/smplx_uv.obj'
    smpl_uv_mesh = trimesh.load_mesh(path2uv_mesh)

    for path2image in data_all:
        dir_name, filename = path2image[0].split('/')[-2:]
        image = cv2.imread(path2image[0], cv2.IMREAD_ANYCOLOR)
        dense_uv = densepose_predictor.get_dense_uv(image)

        image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dense_uv_ = cv2.cvtColor(dense_uv, cv2.COLOR_RGB2BGR)
        input_data = dataset.load_in_the_wild(image_, dense_uv_)

        # unet-based prediction
        input_data["image_target"] = model_pre.in_the_wild_step(input_data)

        # pred noise
        pred_noise, noise, t = model.ddpm_from_noise(input_data, return_extra=True)

        # denoise
        denoised_latent = model.denoise(noise, pred_noise, t)
        denoised_img = model.decode_image_from_latents(denoised_latent)
        denoised = denoised_img[0].detach().cpu().numpy() * 255.0
        denoised = cv2.cvtColor(denoised.transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, filename[:-4]+"_input.png"), image)
        cv2.imwrite(os.path.join(dir_name, filename[:-4]+"_dense_uv.png"), dense_uv)

        mesh = TexturedMesh(smpl_uv_mesh.vertices, smpl_uv_mesh.faces, smpl_uv_mesh.visual.uv, tex=denoised)
        mesh_out = mesh.to_trimesh(with_texture=True, flip=False)
        mesh_out.export(os.path.join(dir_name, filename[:-4]+"_mesh.obj"))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn', force=True)
    # set "highest", "high", or "medium
    torch.set_float32_matmul_precision("medium")
    main()
