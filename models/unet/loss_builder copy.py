import cv2
import copy
import random
import torch
import yaml
import trimesh
import smplx
import json
import torch.nn as nn
import numpy as np
import torchvision
import nvdiffrast.torch as dr
from pytorch_msssim import SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diff_renderer.diff_optimizer import DiffOptimizer
from diff_renderer.diff_optimizer import Renderer
from diff_renderer.normal_nds.nds.core.mesh_ext import TexturedMesh
from diff_renderer.normal_nds.nds.core.mesh_smpl import SMPLMesh
from smpl_optimizer.smpl_wrapper import BaseWrapper
from utils.visualizer import to_trimesh
from PIL import Image
import torchvision.transforms as transforms

def get_plane_params(z, xy, pred_res=512, real_dist=220.0,
                     z_real=False, v_norm=False):
    def gradient_x(img):
        img = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        img = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def normal_from_grad(grad_x, grad_y):
        if pred_res == 512:
            scale = 4.0
        elif pred_res == 1024:
            scale = 8.0
        elif pred_res == 2160:
            scale = 16.0

        grad_z = torch.ones_like(grad_x).float() / scale  # scaling factor (to magnify the normal)
        n = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + torch.pow(grad_z, 2))
        normal = torch.cat((grad_y / n, grad_x / n, grad_z / n), dim=1)
        return normal

    if z is None:
        return None

    if len(z.shape) == 3:
        z = z.unsqueeze(1)

    if z_real:  # convert z to real, if this is set to True
        z = (z - 0.5) * 128.0 + real_dist

    grad_x = gradient_x(z)
    grad_y = gradient_y(z)

    mask = torch.zeros_like(z)
    mask[z > 50] = 1.0
    n_ = normal_from_grad(grad_x, grad_y)
    xyz = torch.cat([xy * z, z], dim=1)
    d = torch.sum(n_ * xyz, dim=1, keepdim=True)
    plane = torch.cat((n_, d), dim=1) * mask

    if v_norm:
        # from [-1, 1] to be in [0, 1];
        # not related to normal_from_grad
        plane[:, 0:3, :, :] += 1
        plane[:, 0:3, :, :] /= 2.0
        plane[:, 3, :, :] /= 255.0

    return plane
# for all networks
class LossBank:
    def __init__(self):  # set loss criteria and options
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.huber_loss = nn.SmoothL1Loss()
        self.lpip_loss = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.criterion_vgg = VGGPerceptualLoss()
        self.criterion_cos = nn.CosineSimilarity(dim=1)
        self.ssim_ch1 = SSIM(data_range=1.0, size_average=True,
                             nonnegative_ssim=True, channel=1, win_size=5)
        self.ssim_ch3 = SSIM(data_range=1.0, size_average=True,
                             nonnegative_ssim=True, channel=3, win_size=5)
        # self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    # l1 loss
    def get_l1_loss(self, pred, target):
        loss = self.l1_loss(pred, target)
        return loss

    # Huber loss
    def get_huber_loss(self, pred, target):
        loss = self.huber_loss(pred, target)
        return loss

    # l1 loss
    def get_l2_loss(self, pred, target):
        loss = self.l2_loss(pred, target)
        return loss

    # binary cross entropy
    def get_bce_loss(self, pred, target):
        loss = self.bce_loss(pred, target)
        return loss

    def get_smoothness_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def get_cosine_loss(self, pred, target):
        # inputs are in [0, 1] -> mapping to [-1, 1]
        loss = 1 - self.criterion_cos(pred * 2 - 1, target * 2 - 1).mean()
        return loss

    # def get_perceptual_loss(self, pred, target):
    #     # to be in [-1, 1]
    #     self.lpip_loss = self.lpip_loss.to(pred.device)
    #     loss = self.lpip_loss(pred * 2.0 - 1.0, target * 2.0 - 1.0)
    #     return loss

    def get_perceptual_loss(self, pred, target):
        loss = self.criterion_vgg(pred, target)
        return loss
    # def get_ms_ssim_loss(self, pred, target):
    #     return self.ms_ssim(pred, target)

    def get_ssim_loss(self, pred, target):
        if pred.shape[1] == 1:
            ssim_loss = 1 - self.ssim_ch1(pred, target)
        else:
            ssim_loss = 1 - self.ssim_ch3(pred, target)
        return ssim_loss

# added on May 10
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        device = input.device
        input = (input - self.mean.to(device)) / self.std.to(device)
        target = (target - self.mean.to(device)) / self.std.to(device)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            block.to(device)
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class LossBuilderHumanUV(LossBank):
    def __init__(self, res=512, weight_conf=1, opt=None, device='cuda:0'):
        LossBank.__init__(self)
        self.device = device
        self.res = res
        self.dr_loss = opt.data.dr_loss
        self.weight_conf = weight_conf
        self.RGB_MEAN = torch.Tensor([0.485, 0.456, 0.406]) # vgg mean
        self.RGB_STD = torch.Tensor([0.229, 0.224, 0.225]) # vgg std
        self.resize_transform = transforms.Resize(self.res)
        # path2config = '/mnt/DATASET8T/home/jumi/Workspace/simple_diffusion/config/exp_config.yaml'
        path2config = '/home/recon/Workspace/uv_recon_ces/config/exp_config.yaml'
        with open(path2config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.params = params
        self.diffopt = DiffOptimizer(params, device=self.device)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.render = Renderer(params['RENDER'], device=self.device)
        self.render_options_gt = {'color': True,
                                  'depth': False,
                                  'normal': True,
                                  'mask': False,
                                  'offset': False}
        self.render_options_smpl = {'color': True,
                                    'depth': False,
                                    'normal': True,
                                    'mask': False,
                                    'offset': False}
        self.use_opengl = True
        self.glctx = dr.RasterizeGLContext() if self.use_opengl else dr.RasterizeCudaContext()
        self.cameras = self.diffopt.set_cams_from_angles(interval_v=opt.data.v_interval,
                                                         interval_h=opt.data.h_interval)
        self.smpl_mesh = None

    def forward(self, pred_var, target_var):
        total_loss = 0.0
        output_log = dict()
        w = [0.9, 0.1]
        w_dr = [0.8, 0.2]
        tgt_uv = target_var['uv_target']
        render_pred_img = []
        render_pred_normal = []

        pred_uv = pred_var['uv']
        loss_uv = w[0] * self.get_loss(pred_uv, tgt_uv, loss_type='l1')
        loss_uv += w[1] * self.get_loss(pred_uv, tgt_uv, loss_type='vgg')
        total_loss += loss_uv
        output_log['uv_pred'] = pred_uv
        output_log['uv_tgt'] = tgt_uv

        if 'disp' in pred_var:
            tgt_disp = target_var['disp_target']
            permute = [2, 1, 0]
            pred_disp = torch.zeros_like(tgt_disp)
            pred_uv = torch.zeros_like(tgt_disp)
            gt_uv = torch.zeros_like(tgt_disp)

            for bat in range(pred_var['disp'].shape[0]):
                pred_disp_ = self.up_sample(pred_var['disp'][bat][None, :, :, :])
                pred_uv_ = self.up_sample(pred_var['uv'][bat][None, :, :, :])
                gt_uv_ = self.up_sample(tgt_uv[bat][None, :, :, :])
                disp = self.diffopt.uv_sampling(pred_disp_[0].permute(1, 2, 0), target_var['smpl_uv_vts'][bat])
                uv = self.diffopt.uv_sampling((torch.flipud(pred_uv_[0].permute(1, 2, 0)))[:, :, permute], target_var['smpl_uv_vts'][bat])
                uv_gt = self.diffopt.uv_sampling((torch.flipud(gt_uv_[0].permute(1, 2, 0)))[:, :, permute], target_var['smpl_uv_vts'][bat])
                disp = disp.permute(1, 0)[None, :, :]
                uv = uv.permute(1, 0)[None, :, :]
                uv_gt = uv_gt.permute(1, 0)[None, :, :]
                pred_disp[bat] = disp
                pred_uv[bat] = uv
                gt_uv[bat] = uv_gt

                if self.dr_loss:
                    tex_color = torch.flipud((pred_uv_[0].detach().cpu() * self.RGB_STD.view(3, 1, 1)
                                              + self.RGB_MEAN.view(3, 1, 1)).permute(1, 2, 0)).contiguous()
                    # mesh_pred = TexturedMesh((target_var['smpl_vertices'][bat] + pred_disp[bat].permute(1, 0)),
                    #                          target_var['smpl_faces'][bat],
                    #                          target_var['smpl_uv_vts'][bat],
                    #                          tex_color * 255,
                    #                          device=self.device) # pred_canon=True -> canonical mesh
                    smpl_params = BaseWrapper.load_params(target_var["smpl_param_path"][bat])
                    smplx_model = smplx.create(
                        model_path='/home/recon/Workspace/uv_recon_ces/resource/smpl_models',
                        model_type='smplx',
                        gender=smpl_params['gender'],
                        num_betas=10,
                        ext='npz',
                        use_face_contour=True,
                        flat_hand_mean=True,
                        use_pca=True,
                        use_pca_comps=6).to(self.device)
                    scan_smpl_mesh = SMPLMesh(vertices=(target_var['smpl_vertices'][bat]
                                                        + pred_disp[bat].permute(1, 0)) / 100,
                                              indices=target_var['smpl_faces'][bat],
                                              uv_vts=target_var['smpl_uv_vts'][bat])
                    # canonical_mesh = scan_smpl_mesh.to_trimesh(with_texture=False)
                    scan_smpl_mesh.smpl_model = smplx_model
                    scan_smpl_mesh.lbs_weights = target_var['lbs_weights'][bat]
                    scan_smpl_mesh.A = target_var['A'][bat]
                    v_deformed = scan_smpl_mesh.forward_skinning(smpl_params, scan_smpl_mesh.vertices[None, :, :])

                    # visual = trimesh.visual.TextureVisuals(uv=target_var['smpl_uv_vts'][bat].detach().cpu().numpy(), image=None)
                    # deformed_smplx = trimesh.Trimesh(vertices=v_deformed.detach().cpu().numpy(),
                    #                                  faces=target_var['smpl_faces'][bat].detach().cpu().numpy(),
                    #                                  visual=visual,
                    #                                  vertex_colors=(pred_uv[bat].permute(1, 0).detach().cpu().numpy())*self.RGB_STD+self.RGB_MEAN)
                    # deformed_smplx.export('pred.obj')

                    mesh_pred = TexturedMesh(v_deformed,
                                             target_var['smpl_faces'][bat],
                                             target_var['smpl_uv_vts'][bat],
                                             tex_color * 255,
                                             device=self.device) # pred_canon=True -> canonical mesh
                    # mesh_canon = TexturedMesh(scan_smpl_mesh.vertices * 100,
                    #                          target_var['smpl_faces'][bat],
                    #                          target_var['smpl_uv_vts'][bat],
                    #                          tex_color * 255,
                    #                          device=self.device) # pred_canon=True -> canonical mesh
                    # canon_mesh\
                    #     = TexturedMesh.to_trimesh(mesh_canon)
                    # canon_mesh.export('canon.obj')
                    pred_mesh = TexturedMesh.to_trimesh(mesh_pred)
                    pred_mesh.export('pred.obj')

                    render_pred = []
                    for camera in self.cameras:
                        render_pred.append(self.render.render(self.glctx, mesh_pred, camera,
                                                              self.render_options_smpl,
                                                              resolution=int(self.cameras[0].K[0,2]*2)))


                    for v in range(len(self.cameras)):
                        render_color = render_pred[v]["color"][0].permute(2, 0, 1)
                        render_normal = render_pred[v]["normal"].permute(2, 0, 1)

                        if render_color.shape[1] is not self.res:
                            render_color = self.resize_transform(render_color)
                            render_normal = self.resize_transform(render_normal)
                            render_pred_img.append(render_color)
                            render_pred_normal.append(render_normal)

                        total_loss += w_dr[0] * self.get_loss(render_color.to(target_var['dr_imgs_target'][v][bat].device),
                                                              target_var['dr_imgs_target'][v][bat], loss_type='l1')
                        total_loss += w_dr[1] * self.get_loss(render_color[None, :, :, :].to(target_var['dr_imgs_target'][v][bat].device),
                                                              target_var['dr_imgs_target'][v][bat][None, :, :, :],
                                                              loss_type='vgg')

                        total_loss += w_dr[0] * self.get_loss(render_normal.to(target_var['dr_normals_target'][v][bat].device),
                                                              target_var['dr_normals_target'][v][bat], loss_type='l2')
                        total_loss += w_dr[1] * self.get_loss(render_normal[None, :, :, :].to(target_var['dr_normals_target'][v][bat].device),
                                                              target_var['dr_normals_target'][v][bat].unsqueeze(0),
                                                              loss_type='cos')

            output_log['render_pred_img'] = render_pred_img
            output_log['render_pred_normal'] = render_pred_normal
            output_log['render_tgt_img'] = target_var['dr_imgs_target']
            output_log['render_tgt_normal'] = target_var['dr_normals_target']

            total_loss += w[0] * self.get_loss(pred_disp, tgt_disp, loss_type='l2')
            total_loss += w[1] * self.get_loss(pred_disp, tgt_disp, loss_type='cos')

            # test (reconstruct mesh)
            # pred_mesh = trimesh.Trimesh(vertices=(target_var['smpl_vertices'][0] + pred_disp[0].permute(1, 0)).detach().cpu().numpy(),
            #                             faces=target_var['smpl_faces'][0].detach().cpu().numpy(),
            #                             vertex_colors=(pred_uv[0].permute(1, 0).detach().cpu().numpy())*self.RGB_STD+self.RGB_MEAN)
            # gt_mesh = trimesh.Trimesh(vertices=(target_var['smpl_vertices'][0]  + tgt_disp[0].permute(1, 0)).detach().cpu().numpy(),
            #                           faces=target_var['smpl_faces'][0].detach().cpu().numpy(),
            #                           vertex_colors=(gt_uv[0].permute(1, 0).detach().cpu().numpy())*self.RGB_STD+self.RGB_MEAN)
            # pred_mesh.export('pred.obj')
            # gt_mesh.export('gt.obj')

        return total_loss, output_log

    # custom loss functions here.
    def get_loss(self, pred, target, loss_type='l1', sigma=None, weight=0.1):
        if loss_type == 'l1':
            loss = self.get_l1_loss(pred, target)
        elif loss_type == 'bce':
            loss = self.get_bce_loss(pred, target)
        elif loss_type == 'l2' or loss_type == 'mse':
            loss = self.get_l2_loss(pred, target)
        elif loss_type == 'vgg':
            loss = self.get_perceptual_loss(pred, target)
        elif loss_type == 'ssim':
            loss = self.get_ssim_loss(pred, target)
        elif loss_type == 'ms_ssim':
            loss = self.get_ms_ssim_loss(pred, target)
        elif loss_type == 'cos':
            loss = self.get_cosine_loss(pred, target)
        elif loss_type == 'sigma' and sigma is not None:  # laplacian loss.
            loss = self.get_laplacian_loss(pred, target, sigma, weight=weight)
        elif loss_type == 'smooth' or loss_type == 'smoothness':
            loss = self.get_smoothness_loss(pred, target)
        else:
            loss = self.get_l1_loss(pred, target)
        return loss

if __name__ == '__main__':
    # arr = np.arange(1, 10)
    a = [1, 2, 3]
    b = [4, 5, 6]
    for k, p in enumerate(a):
        print(k)
    # print(arr)
