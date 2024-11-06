import numpy as np
import torch
import torch.nn as nn


def gradient_x(img):
    img = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    img = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy


def get_plane_params(z, xy, pred_res=512, real_dist=220.0,
                     z_real=False, v_norm=False):
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
    # for loss (predict normalized values)
    # (1) to normalize plane parameters.
    # plane[:, 0:3, :, :] += 1
    # plane[:, 0:3, :, :] /= 2
    # plane[:, 3, :, :] /= 255
    # (2) to denormalize plane parameters.
    # plane[:, 0:3, :, :] *= 2
    # plane[:, 0:3, :, :] -= 1
    # plane[:, 3, :, :] *= 255
    # (3) unit normal loss
    # loss_norm = torch.mean(torch.norm(plane[:, 0:3, :, :]) - 1)
    # (4) depth loss
    # x = np.reshape((np.linspace(0, w, w) - cam.principal_x)/cam.focal_x, [1, 1, -1, 1])
    # x = np.tile(x, [1, 1, 1, h])
    # y = np.reshape((np.linspace(0, h, h) - cam.principal_y)/cam.focal_y, [1, 1, 1, -1])
    # y = np.tile(y, [1, 1, w, 1])
    # xy = torch.Tensor(np.concatenate((x, y), axis=1))
    # z = d - torch.sum( xy * plane[:, 0:2, :, :], dim=1 )


def depth2normal(img, normalize=True):
    zy, zx = np.gradient(img.squeeze())
    # normal = np.dstack((-zx, -zy, np.ones_like(np.array(img))))
    normal = np.dstack((zx, zy, np.ones_like(np.array(img))))

    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    if normalize is not True:
        normal *= 255

    return normal


def depth2normal_torch(z, normalize=True):
    # input: B x 1 x W X H
    # output: B x 3 x W x H
    B, C, _, _ = z.size()
    g_x = gradient_x(z)
    g_y = gradient_y(z)
    g_z = torch.ones_like(g_x)

    n = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + g_z)
    normal = torch.cat((g_x / n, g_y / n, g_z / n), dim=1)
    normal += 1
    normal /= 2

    if normalize is not True:
        normal *= 255
    return normal


def gradient_torch(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x
