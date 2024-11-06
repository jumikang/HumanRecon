import os

import cv2
import numpy as np
import trimesh
import torch
import glob
from PIL import Image
import collections
import grouping

def load_textured_mesh(path2mesh, image_only=False):
    """
    Load textured mesh
    Initially, find filename.bmp/jpg/tif/jpeg/png as a texture map.
     > if there are multiple images, take the first one.
     > otherwise, find material_0.bmp/jpe/tif/jpeg/png as a texture map.
    :param path2mesh: path to the textured mesh (.obj file only)
    :param filename: name of the current mesh
    :return: mesh with texture (texture will not be defined, if the texture map does not exist)
    """
    filename = path2mesh.split('/')[-1][:-4]

    exts = ['.tif', '.bmp', '.jpg', '.jpeg', '.png', '_0.png']
    # text_file = os.path.join(self.path2obj, filename, filename)
    text_file = [path2mesh.replace('.obj', ext) for ext in exts
                 if os.path.isfile(path2mesh.replace('.obj', ext))]
    if len(text_file) == 0:
        # text_file = os.path.join(self.path2obj, filename, 'material_0')
        text_file = path2mesh.replace(filename + '.obj', 'material_0')
        text_file = [text_file + ext for ext in exts if os.path.isfile(text_file + ext)]

    # for RP_T dataset
    if len(text_file) == 0:
        obj = path2mesh.split('/')[-1]
        text_file = os.path.join(path2mesh.replace(obj, ''), 'tex', filename.replace('FBX', 'dif.jpg'))
        if os.path.isfile(text_file):
            text_file = [text_file]
    if len(text_file) > 0:
        im = Image.open(text_file[0])
        texture_image = np.array(im)
    else:
        texture_image = None

    if image_only:
        return texture_image
    else:
        mesh = trimesh.load_mesh(path2mesh, process=False)  # normal mesh
        return mesh, texture_image

def load_gt_data(path2obj, height=180.0):
    """
    NOT OPTIMIZED YET
    :param data:
    :param height:
    :return:
    """
    # input data loading
    obj_path = os.path.join(path2obj)

    if len(obj_path) > 0:
        if isinstance(obj_path, list):
            obj_path = obj_path[0]

        m, texture_image = load_textured_mesh(obj_path)
        if texture_image is None:
            print('could not find texture map')
        else:
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            texture_image = np.rot90(texture_image, k=1)
            texture_image = np.flip(texture_image, axis=1)
            texture_image = np.rot90(texture_image, k=3)

        # for standing mesh (general)
        vertices = m.vertices
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        up_axis = 1
        center = np.median(vertices, 0)
        center[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

        scale = height / (vmax[up_axis] - vmin[up_axis])

        # without library (comment below lines for non-agisoft results)

        # vertices, faces, textures, face_textures = load_file2info(obj_path)
        vertices = (vertices - center) * scale
        mesh = trimesh.load_mesh(path2obj, process=False)
        return vertices, mesh.faces, mesh.visual.uv, mesh.faces, texture_image

# def load_file2info(mesh_file):
#     vertex_data = []
#     norm_data = []
#     uv_data = []
#     dict = collections.defaultdict(int)
#
#     face_data = []
#     face_norm_data = []
#     face_uv_data = []
#
#     if isinstance(mesh_file, str):
#         f = open(mesh_file, "r")
#     else:
#         f = mesh_file
#
#     for line in f:
#         if isinstance(line, bytes):
#             line = line.decode("utf-8")
#         if line.startswith('#'):
#             continue
#         values = line.split()
#         if not values:
#             continue
#         if values[0] == 'v':
#             v = list(map(float, values[1:4]))
#             vertex_data.append(v)
#         elif values[0] == 'vn':
#             vn = list(map(float, values[1:4]))
#             norm_data.append(vn)
#         elif values[0] == 'vt':
#             vt = list(map(float, values[1:3]))
#             uv_data.append(vt)
#         elif values[0] == 'f':
#             # quad mesh
#             if len(values) > 4:
#                 f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
#                 face_data.append(f)
#                 f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
#                 face_data.append(f)
#             # tri mesh
#             else:
#                 f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
#                 face_data.append(f)
#
#             # deal with texture
#             if len(values[1].split('/')) >= 2:
#                 # quad mesh
#                 if len(values) > 4:
#                     f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
#                     face_uv_data.append(f)
#                     f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
#                     face_uv_data.append(f)
#                 # tri mesh
#                 elif len(values[1].split('/')[1]) != 0:
#                     f_c = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
#                     face_uv_data.append(f_c)
#                     f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
#                     dict[f[0] - 1] = f_c[0] - 1
#                     dict[f[1] - 1] = f_c[1] - 1
#                     dict[f[2] - 1] = f_c[2] - 1
#                 else:
#                     face_uv_data.append([1, 1, 1])
#
#             # deal with normal
#             if len(values[1].split('/')) == 3:
#                 # quad mesh
#                 if len(values) > 4:
#                     f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
#                     face_norm_data.append(f)
#                     f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
#                     face_norm_data.append(f)
#                 # tri mesh
#                 elif len(values[1].split('/')[2]) != 0:
#                     f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
#                     face_norm_data.append(f)
#
#     vertex_colors = []
#     for k in range(len(vertex_data)):
#         if k in dict:
#             vertex_colors.append(uv_data[dict[k]])
#         else:
#             vertex_colors.append([0.0, 0.0])
#
#     vertices = np.array(vertex_data)
#     uvs = np.array(vertex_colors)
#     faces = np.array(face_data) - 1
#     face_uvs = np.array(face_uv_data) - 1
#     # vertices, faces, mid_uvs = subdivide_keti(vertices, uv_data, faces, face_uvs)
#     # uvs = np.vstack((visuals, mid_uvs))
#
#     return vertices, faces, uvs, face_uvs

def load_file2info(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, list):
        mesh_file = mesh_file[0]

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data) - 1
    return vertices, faces, uvs, face_uvs