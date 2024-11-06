import collections
import json
import os
import torch
import numpy as np
import random
import cv2
import trimesh
from torch.nn.functional import smooth_l1_loss

import nvdiffrast.torch as dr
from tqdm import tqdm
from torch import nn

# from apps.hand_replace import smpl_mesh
from diff_renderer.normal_nds.nds.core import Camera
from diff_renderer.normal_nds.nds.core.mesh_ext import TexturedMesh
from diff_renderer.normal_nds.nds.losses import laplacian_loss


class Renderer:
    def __init__(self, params, near=1, far=1000, orthographic=False, device='cuda'):

        self.max_mip_level = params['max_mip_level']
        self.angle_interval = params['angles']

        self.res = 1024

        # self.glctx = dr.RasterizeGLContext()
        self.device = device
        self.near = near
        self.far = far
        self.orthographic = orthographic

    def set_near_far(self, views, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
        """
        mins = []
        maxs = []
        for view in views:
            samples_projected = view.project(samples, depth_as_distance=True)
            mins.append(samples_projected[...,2].min())
            maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)
        self.near = near - (near * epsilon)
        self.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device)
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000, orthographic=False):
        if orthographic:
            projection_matrix = torch.eye(4, device=camera.device)
            projection_matrix[:3, :3] = camera.K
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  -1., 0,  0],
                                        [0,  0, -1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)
        else:
            projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                    fy=camera.K[1,1],
                                                    cx=camera.K[0,2],
                                                    cy=camera.K[1,2],
                                                    n=n,
                                                    f=f,
                                                    width=resolution[1],
                                                    height=resolution[0],
                                                    device=camera.device)
            gl_transform = torch.tensor([[1., 0,  0,  0],
                                        [0,  -1., 0,  0],
                                        [0,  0, 1., 0],
                                        [0,  0,  0,  1.]], device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    def get_gl_camera(self, camera, resolution):
        P = self.to_gl_camera(camera, resolution, n=self.near, f=self.far, orthographic=self.orthographic)
        return P

    def render(self, glctx, mesh, camera, render_options,
               resolution=1024,
               verts_init=None,
               enable_mip=True):

        render_out = {}

        def transform_pos(mtx, pos):
            t_mtx = torch.from_numpy(mtx) if isinstance(mtx, np.ndarray) else mtx
            posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)
            return torch.matmul(posw, t_mtx.t().to(pos.device))[None, ...]

        pos, pos_idx, uv, tex, disp = \
            mesh.vertices, mesh.indices.int(), mesh.uv_vts, mesh.tex, mesh.disp

        mtx = self.to_gl_camera(camera, [resolution, resolution], n=self.near, f=self.far, orthographic=self.orthographic)
        pos_clip = transform_pos(mtx, pos)
        rast, rast_db = dr.rasterize(glctx, pos_clip.type(torch.float32), pos_idx, resolution=[resolution, resolution])

        if render_options["color"]:
            if enable_mip:
                texc, texd = dr.interpolate(uv[None, ...], rast, pos_idx, rast_db=rast_db, diff_attrs='all')
                color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear',
                                   max_mip_level=self.max_mip_level)
            else:
                texc, _ = dr.interpolate(uv[None, ...], rast, pos_idx)
                color = dr.texture(tex[None, ...], texc, filter_mode='linear')[0]
            render_out["color"] = color * torch.clamp(rast[..., -1:], 0, 1)  # Mask out background.

        if render_options["mask"]:
            mask = torch.clamp(rast[..., -1:], 0, 1)
            mask = dr.antialias(mask, rast, pos_clip, pos_idx)[0] # if with_antialiasing else mask[0]
            render_out["mask"] = mask

        if render_options["normal"]:
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, pos_idx)
            render_out["normal"] = dr.antialias(normal.type(torch.float32), rast.type(torch.float32), 
                                                pos_clip.type(torch.float32), pos_idx)[0] # if with_antialiasing else normal[0]

        if render_options["depth"]:
            position, _ = dr.interpolate(pos[None, ...], rast, pos_idx)
            render_out["depth"] = dr.antialias(position, rast, pos_clip, pos_idx)[0] # if with_antialiasing else position[0]
            # gbuffer["depth"] = view.project(gbuffer["position"], depth_as_distance=True)[..., 2:3]

        # for future use
        if render_options["offset"]:
            texc, _ = dr.interpolate(uv[None, ...], rast, pos_idx)
            render_out["disp_uv"] = dr.texture(disp[None, ...], texc, filter_mode='linear')[0]

            position, _ = dr.interpolate(pos[None, ...] - verts_init[None, ...], rast, pos_idx)
            render_out["disp_cv"] = position[0]

        return render_out  # texture map only

    def get_vert_visibility(self, glctx, mesh, camera, resolution=1024):
        vertices = mesh.vertices
        idx = mesh.indices.int()
        num_verts = len(vertices)

        with torch.no_grad():
            # for camera in cameras:
            vis_mask = torch.zeros(size=(num_verts,), device=self.device).bool()  # visibility mask
            P = Renderer.to_gl_camera(camera, [resolution, resolution],
                                      n=self.near, f=self.far,
                                      orthographic=self.orthographic)
            pos = Renderer.transform_pos(P, vertices)
            rast, rast_out_db = dr.rasterize(glctx, pos, idx, resolution=np.array([resolution, resolution]))

            # Do not support batch operation yet
            face_ids = rast[..., -1].long()
            masked_face_idxs_list = face_ids[face_ids != 0] - 1  # num_masked_face Tensor
            # masked_face_idxs_all = torch.unique(torch.cat(masked_face_idxs_list, dim=0))
            masked_verts_idxs = torch.unique(idx[masked_face_idxs_list].long())
            vis_mask[masked_verts_idxs] = 1
            vis_mask = vis_mask.bool().to(self.device)
            # vis_masks.append(vis_mask)
        return vis_mask


class DiffOptimizer(nn.Module):
    def __init__(self,
                 params,
                 init_uv=None,
                 device='cuda:0'):
        super(DiffOptimizer, self).__init__()

        self.cam_params = params['CAM']
        self.render_res = 1024
        self.use_opengl = False
        self.device = device

        path2label = os.path.join(params['SMPL']['smpl_root'], params['SMPL']['segmentation'])
        with open(path2label, 'r') as f:
            self.v_label = json.load(f)
        path2mapper = os.path.join(params['SMPL']['smpl_root'], params['SMPL']['uv_mapper'])
        with open(path2mapper, 'r') as f:
            self.uv_mapper = json.load(f)

        # subdivision is possible only when init_uv is given!
        self.seam, self.seam_uv = [], []
        self.compute_seam(uv_vts=init_uv)

        self.render = Renderer(params['RENDER'], device=device)

    def set_cams_from_angles(self, interval_v=[0], interval_h=[0], device='cuda:0'):
        cameras = []
        for v in interval_v:
            for h in interval_h:
                camera = Camera.perspective_camera_with_angle(view_angle=h, pitch=v, cam_params=self.cam_params, device=device)
                cameras.append(camera)
        return cameras
    
    def set_cams_random_angles(self, angle_h=None, angle_v=None, camera_num=1):
        cameras = []
        for i in range(camera_num):
            camera = Camera.perspective_camera_with_angle(view_angle=angle_h[i],
                                                          pitch=angle_v[i],
                                                          cam_params=self.cam_params)
            cameras.append(camera)
        return cameras
    
    def set_cams_from_random_angles(self, cam_nums=1):
        cameras = []
        for _ in range(cam_nums):
            angle_h = random.randint(0, 359)
            angle_v = random.randint(-10, 10)
            camera = Camera.perspective_camera_with_angle(view_angle=angle_h, pitch=angle_v, cam_params=self.cam_params)
            cameras.append(camera)
        return cameras

    def uv_sampling(self, tex, uv):
        '''

        :param displacement: [B, C, H, W] image features
        :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
        :return: [B, C, N] image features at the uv coordinates
        '''
        # uv is in [0, 1] so we need to convert to be in [-1, 1]
        uv = uv.unsqueeze(0).unsqueeze(2) * 2 - 1
        tex = tex.permute(2, 0, 1).unsqueeze(0)
        samples = torch.nn.functional.grid_sample(tex, uv, align_corners=True)  # [B, C, N, 1]
        return samples.squeeze().transpose(1, 0).contiguous()

    def pipeline(self, mesh_smpl, mesh_gt):
        # initial uv unwrapping.
        # weights = {'color': 1.0, 'depth': 1.0, 'disp': 100.0, 'normal': 1.0,
        #            'seam': 100.0, 'smooth': 1.0, 'laplacian': 100.0}
        # mesh_smpl = self(mesh_smpl, mesh_gt, weights=weights, max_iter=1000, lr=1e-2)

        # refined unwrapping #1.
        weights = {'color': 1.0, 'depth': 1.0, 'disp': 0.1, 'normal': 1.0,
                   'seam': 100.0, 'smooth': 1.0, 'laplacian': 10.0}
        # mesh_smpl.upsample()
        # self.update_seam(mesh_smpl.uv_vts)
        mesh_smpl = self(mesh_smpl, mesh_gt, weights=weights, max_iter=1000, lr=1e-2)

        # refined unwrapping #2.
        # weights = {'color': 1.0, 'depth': 1.0, 'disp': 0.1, 'normal': 1.0,
        #            'seam': 100.0, 'smooth': 1.0, 'laplacian': 1.0}
        # mesh_smpl.upsample()
        # self.update_seam(mesh_smpl.uv_vts)
        # mesh_smpl = self(mesh_smpl, mesh_gt, weights=weights, max_iter=1000, lr=1e-2)

        deformed_mesh = mesh_smpl.to_trimesh(with_texture=True)
        # deformed_mesh.show()
        deformed_mesh.export('deformed.obj')
        return deformed_mesh

    @staticmethod
    def tv_loss(img):
        xv = img[1:, :, :] - img[:-1, :, :]
        yv = img[:, 1:, :] - img[:, :-1, :]
        loss = torch.mean(abs(xv)) + torch.mean(abs(yv))
        return loss

    def compute_seam(self, uv_vts=None):
        """
        compute indices for seam
        """
        tmp = collections.defaultdict(int)
        for i, n in enumerate(self.uv_mapper['smplx_uv']):
            if n not in tmp:
                tmp[n] = [i]
            else:
                tmp[n].append(i)

        seam = []
        for key in tmp.keys():
            if len(tmp[key]) > 1:
                seam.append(tmp[key])

        self.seam = [[s[0], s[1]] for s in seam]
        for s in seam:
            if len(s) == 3:
                self.seam.append([s[1], s[2]])
            if len(s) == 4:
                self.seam.append([s[2], s[3]])
        self.seam = np.asarray(self.seam)
        if uv_vts is not None:
            self.seam_uv = dict()
            for k in range(self.seam.shape[0]):
                self.seam_uv[tuple(uv_vts[self.seam[k, 0], :])] = tuple(uv_vts[self.seam[k, 1], :])

    def update_seam(self, new_vts):
        uv_vts = new_vts.detach().cpu().numpy().tolist()
        uv_vts = [tuple(np.round(uv, 7)) for uv in uv_vts]
        new_seam = []
        for k in range(len(uv_vts)):
            key = tuple(uv_vts[k])
            if key in self.seam_uv:
                j = uv_vts.index(self.seam_uv[key])
                new_seam.append([k, j])

        self.seam = np.asarray(new_seam)

    # update geometry, color, or both
    def forward(self, mesh_smpl, mesh_scan,
                weights=None, max_iter=10000, lr=1e-2):

        texture_opt = mesh_smpl.tex
        texture_opt.retain_grad()

        disp_opt = mesh_smpl.disp
        disp_opt.requires_grad_()
        disp_opt.retain_grad()

        vertex_initial = mesh_smpl.vertices.clone()

        opt_params = [texture_opt, disp_opt]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

        glctx = dr.RasterizeGLContext() if self.use_opengl else dr.RasterizeCudaContext()
        cameras = self.set_cams_from_angles(interval=10) # degree

        render_options = {'color': True,
                          'depth': True,
                          'normal': True,
                          'mask': False,
                          'offset': False}
        render_tgt = []
        for camera in cameras:
            render_tgt.append(self.render.render(glctx, mesh_scan, camera, render_options))

        view_num = len(cameras)
        log_interval = 10
        scheduler_interval = 100
        l2_loss = nn.MSELoss()

        render_options_src = {'color': True,
                              'depth': True,
                              'normal': True,
                              'mask': False,
                              'offset': True}
        for k in tqdm(range(max_iter), 'opt. mesh:'):
            v = random.randint(0, view_num - 1)
            mesh_smpl.vertices = self.uv_sampling(disp_opt, mesh_smpl.uv_vts) + vertex_initial

            render_src = self.render.render(glctx, mesh_smpl, cameras[v],
                                            render_options_src,
                                            verts_init=vertex_initial)

            loss = l2_loss(render_src["color"], render_tgt[v]["color"]) * weights["color"]
            loss += l2_loss(render_src["depth"], render_tgt[v]["depth"]) * weights["depth"]
            loss += l2_loss(render_src["disp_uv"], render_src["disp_cv"]) * weights["disp"]
            loss += l2_loss(render_src["normal"], render_tgt[v]["normal"]) * weights["normal"]

            # seam loss
            loss += l2_loss(mesh_smpl.vertices[self.seam[:, 0], :],
                            mesh_smpl.vertices[self.seam[:, 1], :]) * weights["seam"]
            loss += self.tv_loss(disp_opt) * weights["smooth"]
            loss += laplacian_loss(mesh_smpl) * weights["laplacian"]

            # things to do
            # hands and eyeball constraints

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (k % scheduler_interval) == 0:
                scheduler.step()

            if (k % log_interval) == 0:
                # result = np.concatenate([render_tgt[v]["color"].squeeze(0).detach().cpu().numpy(),
                #                         render_src["color"].squeeze(0).detach().cpu().numpy()], axis=1)
                # cv2.imshow('rendered', result)
                # result = np.concatenate([render_tgt[v]["color"].squeeze(0).detach().cpu().numpy(),
                #                         render_src["color"].squeeze(0).detach().cpu().numpy()], axis=1)
                # cv2.imshow('rendered', result)

                texture_map = texture_opt.squeeze(0).detach().cpu().numpy()
                # cv2.imshow('texture_map', texture_map)
                texture_map = disp_opt.squeeze(0).detach().cpu().numpy()
                # cv2.imshow('disp_map', texture_map)
                # cv2.waitKey(10)

        mesh_smpl.tex = texture_opt
        mesh_smpl.detach()
        return mesh_smpl
