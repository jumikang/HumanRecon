import cv2
import os
import json
import smplx
import torch
import trimesh
import numpy as np
import collections
from PIL import Image
from diff_renderer.normal_nds.nds.core.mesh_ext import TexturedMesh, to_torch
from .human_animator.animater_utils import batch_rigid_transform


class SMPLMesh(TexturedMesh):
    def __init__(self,
                 vertices=None,
                 indices=None,
                 uv_vts=None,
                 v_posed=None,
                 lbs_weights=None,
                 tex=None,
                 disp=None,
                 smpl_config=None,
                 device='cuda:0'):
        super(TexturedMesh).__init__()
        self.device = device
        self.smpl_config = smpl_config

        # variables that are calculated internally
        self.lbs_weights, self.joints, self.A = None, None, None
        self.uv_mapper = None
        self.seam, self.seam_uv = None, None
        self.face_normals, self.vertex_normals = None, None
        self._edges = None
        self._connected_faces = None
        self._laplacian = None
        self.smpl_model = None

        # initialize a posed mesh
        self.vertices = self.to_torch(vertices, device)
        self.indices = self.to_torch(indices, device)
        if self.indices is not None:
            self.indices = self.indices.type(torch.int64)
        self.uv_vts = self.to_torch(uv_vts, device)

        if tex is not None:
            self.tex = self.to_torch(tex, device)
        else:
            self.tex = torch.ones((1024, 1024, 3),
                                  dtype=torch.float32,
                                  requires_grad=True).to(device) * 0.2

        if v_posed is not None:
            self.v_posed = self.to_torch(v_posed, device)

        if lbs_weights is not None:
            self.lbs_weights = self.to_torch(lbs_weights, device)

        # displacement map (for optimization purpose, used internally)
        if disp is not None:
            self.disp = self.to_torch(disp, device)
        else:
            self.disp = torch.zeros((1024, 1024, 3),
                                    dtype=torch.float32,
                                    requires_grad=True).to(device)

        if self.indices is not None:
            self.compute_normals()

        if self.smpl_config is not None:
            self.smpl_model = self.init_smpl_model()

    def to_torch(self, data, device, dtype=torch.float32):
        if torch.is_tensor(data):
            torch_tensor = data.to(device, dtype=dtype)
        else:
            torch_tensor = torch.tensor(data.copy(), dtype=dtype, device=device)
        return torch_tensor

    def get_canonical_mesh(self, disp_vectors):
        vertices = self.v_posed + disp_vectors
        vertices = vertices.cpu().detach().numpy()
        uv = self.uv_vts.cpu().detach().numpy()
        faces = self.indices.cpu().detach().numpy().astype(np.int32)

        texture = self.tex.cpu().detach().numpy()
        texture = np.uint8(texture * 255)
        texture_image = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        texture_image = np.rot90(texture_image, k=1)
        texture_image = np.flip(texture_image, axis=1)
        texture_image = np.rot90(texture_image, k=3)
        tex_pil = Image.fromarray(texture_image)

        visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_pil)
        mesh = trimesh.Trimesh(vertices, faces, visual=visual, process=False)
        return mesh

    def init_smpl_model(self):
        """
            create smpl(-x,-h) instance
            :return: a smpl(-x,-h) instance
        """
        return smplx.create(self.smpl_config['smpl_root'],
                            model_type=self.smpl_config['model_type'],
                            gender=self.smpl_config['gender'],
                            num_betas=self.smpl_config['num_betas'],
                            ext=self.smpl_config['ext'],
                            use_face_contour=self.smpl_config['use_face_contour'],
                            flat_hand_mean=self.smpl_config['use_flat_hand'],
                            use_pca=self.smpl_config['use_pca'],
                            num_pca_comps=self.smpl_config['num_pca_comp']
                            ).to(self.device)

    def forward_smpl(self, smpl_params, return_mesh=False):
        smpl_output = self.smpl_model(transl=smpl_params['transl'],
                                      expression=smpl_params['expression'],
                                      body_pose=smpl_params['body_pose'],
                                      betas=smpl_params['betas'],
                                      global_orient=smpl_params['global_orient'],
                                      jaw_pose=smpl_params['jaw_pose'],
                                      left_hand_pose=smpl_params['left_hand_pose'],
                                      right_hand_pose=smpl_params['right_hand_pose'],
                                      return_full_pose=True,
                                      return_verts=True)

        if 'scale' in smpl_params and smpl_params['scale'] is not None:
            smpl_output.joints = smpl_output.joints * smpl_params['scale']
            smpl_output.vertices = smpl_output.vertices * smpl_params['scale']

        if return_mesh:
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy() * 100,
                                        faces=self.smpl_model.faces, process=False)
            return smpl_output, smpl_mesh
        else:
            return smpl_output

    def detach(self):
        mesh = SMPLMesh(vertices=self.vertices.detach(),
                        indices=self.indices.detach(),
                        uv_vts=self.uv_vts.detach() if self.uv_vts is not None else None,
                        v_posed=self.v_posed.detach() if self.v_posed is not None else None,
                        lbs_weights=self.lbs_weights.detach() if self.lbs_weights is not None else None,
                        tex=self.tex.detach() if self.tex is not None else None,
                        disp=self.disp.detach() if self.disp is not None else None,
                        smpl_config=self.smpl_config,
                        device=self.device)

        mesh.A = self.A.detach() if self.A is not None else None
        mesh.joints = self.joints.detach() if self.joints is not None else None
        mesh.seam = self.seam
        mesh.seam_uv = self.seam_uv
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

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
        if torch.is_tensor(new_vts):
            uv_vts = new_vts.detach().cpu().numpy().tolist()
        else:
            uv_vts = new_vts.tolist()

        uv_vts = [tuple(np.round(uv, 7)) for uv in uv_vts]
        new_seam = []
        for k in range(len(uv_vts)):
            key = tuple(uv_vts[k])
            if key in self.seam_uv:
                j = uv_vts.index(self.seam_uv[key])
                new_seam.append([k, j])

        self.seam = np.asarray(new_seam)

    def set_canonical_smpl_in_uv(self, smpl_params):
        # load and set SMPL mesh with uv coordinates.
        path2uv_table = os.path.join(self.smpl_config["smpl_root"], 'smpl_uv_table.json')
        if os.path.exists(path2uv_table):
            with open(path2uv_table, 'r') as f:
                self.uv_mapper = json.load(f)
        path2uv_mesh = os.path.join(self.smpl_config["smpl_root"],
                                    'smplx_uv', 'smplx_uv.obj')
        if os.path.exists(path2uv_mesh) and self.uv_vts is None:
            smpl_uv_mesh = trimesh.load_mesh(path2uv_mesh)
            uv_vts = np.round(smpl_uv_mesh.visual.uv, 7)
            self.uv_vts_np = uv_vts  # for subdivision purpose
            self.uv_vts = self.to_torch(uv_vts, self.device)
            self.indices = self.to_torch(smpl_uv_mesh.faces, self.device).type(torch.int64)
            self.compute_seam(uv_vts)

        else:
            assert "smpl_uv.obj file is necessary"
        for key in smpl_params.keys():
            if torch.is_tensor(smpl_params[key]):
                smpl_params[key] = smpl_params[key].to(self.device)

        smpl_output = self.forward_smpl(smpl_params, return_mesh=False)
        v_shaped = self.smpl_model.v_template + \
                   smplx.lbs.blend_shapes(smpl_output.betas, self.smpl_model.shapedirs)
        self.joints = smplx.lbs.vertices2joints(self.smpl_model.J_regressor, v_shaped)
        self.lbs_weights = self.smpl_model.lbs_weights.to(self.device).detach()

        # do not use smpl_model.joints -> it fails (don't know why)
        batch_size = 1
        rot_mats = smplx.lbs.batch_rodrigues(smpl_output.full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        ident = torch.eye(3, dtype=torch.float32, device=self.device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, self.smpl_model.posedirs).view(batch_size, -1, 3)

        _, A = batch_rigid_transform(rot_mats,
                                     self.joints[:, :55, :],
                                     self.smpl_model.parents,
                                     inverse=False,
                                     dtype=torch.float32)

        v_posed = pose_offsets + v_shaped

        self.lbs_weights = self.lbs_weights[self.uv_mapper['smplx_uv'], :]
        self.v_posed = v_posed.squeeze(0)[self.uv_mapper['smplx_uv'], :]
        self.A = A

    # you can apply this only once.
    def subdivide(self):
        self.detach()
        self.disp_vector = self.uv_sampling(mode='disp')

        vertices, faces = trimesh.remesh.subdivide(
            vertices=np.hstack((self.vertices.detach().cpu().numpy(),
                                self.v_posed.detach().cpu().numpy(),
                                self.uv_vts_np,
                                self.disp_vector.detach().cpu().numpy(),
                                self.lbs_weights.detach().cpu().numpy())),
            faces=self.indices.detach().cpu().numpy())

        # set subdivided smplx mesh with uv coordinates.
        self.vertices = self.to_torch(vertices[:, 0:3], self.device)
        self.v_posed = self.to_torch(vertices[:, 3:6], self.device)
        self.uv_vts = self.to_torch(vertices[:, 6:8], self.device)
        self.disp_vector = self.to_torch(vertices[:, 8:11], self.device)
        self.lbs_weights = self.to_torch(vertices[:, 11:], self.device)
        self.indices = self.to_torch(faces, self.device).type(torch.int64)
        self.update_seam(vertices[:, 6:8])
        self.compute_normals()
        # from diff_renderer.normal_nds.nds.utils.geometry import compute_laplacian_uniform
        # self._laplacian = compute_laplacian_uniform(self)
        # self._edges = self.edges
        # self._connected_faces = self.connected_faces

    def forward_skinning(self, smpl_params=None, v_posed=None, update_smpl=False):
        if v_posed is None:
            v_posed = self.v_posed[None, :, :]

        if update_smpl:
            smpl_output = self.forward_smpl(smpl_params)
            rot_mats = smplx.lbs.batch_rodrigues(smpl_output.full_pose.view(-1, 3)).view([1, -1, 3, 3])
            _, self.A = batch_rigid_transform(rot_mats,
                                              self.joints[:, :55, :],
                                              self.smpl_model.parents,
                                              inverse=False,
                                              dtype=torch.float32)

        weights = self.lbs_weights.expand([1, -1, -1])
        num_joints = self.smpl_model.J_regressor.shape[0]
        T = torch.matmul(weights, self.A.reshape(1, num_joints, 16)).view(1, -1, 4, 4)
        homogen_coord = torch.ones([1, v_posed.shape[1], 1], dtype=torch.float32).to(self.device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T.float().to(self.device), torch.unsqueeze(v_posed_homo, dim=-1))

        # update vertices
        v_deformed = v_homo[:, :, :3, 0]
        if smpl_params is not None:
            # v_deformed = (v_deformed + smpl_params['transl'].to(self.device)) * smpl_params['scale']
            v_deformed = (v_deformed.detach().cpu() + smpl_params['transl']) * smpl_params['scale']
        return v_deformed[0, :, :] * 100.0
