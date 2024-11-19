from __future__ import print_function
# import math
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import pytorch_lightning as pl
# import numpy as np
from models.unet.unet import ATUNet_UV, UNet#, ATUNet, ResUnetPlusPlus
from models.unet.loss_builder import LossBuilderHumanUV
# from models.MSPN_SDR.lib.model.resnet_cbam import resnet101_cbam
# from models.MSPN_SDR.lib.model.NAFNet_arch import NAFNet
# from models.resnet.resnet import ResNetEncoder
# from models.resnet.decoder import Decoder

class BaseModule(nn.Module):
    def __init__(self, im2d_in=6,
                 return_uv=False,
                 return_disp=False,
                 split_last=True):
        super(BaseModule, self).__init__()
        self.return_uv = return_uv
        self.return_disp = return_disp
        self.split_last = split_last
        if self.return_uv and self.return_disp:
            # self.imuv2uvdisp = NAFNet(img_channel=3)
            # self.imuv2uvdisp = ResUnetPlusPlus(channel=6)
            # self.imuv2uvdisp_encoder = ResNetEncoder(in_channels=6)
            # self.imuv2uv_disp_decoder = Decoder(channels=[512, 64, 8, 1, 6],
            #                                     feature_channels=[2048, 256, 64, 6, 7])
            # self.uvFeature = ATUNet_UV(in_ch=im2d_in, out_ch=64, split_last=self.split_last)
            # self.uvFeature2uvd = ATUNet_UV(in_ch=(im2d_in + 64), out_ch=6, split_last=self.split_last)
            self.imuv2uvdisp = ATUNet_UV(in_ch=im2d_in, out_ch=6, split_last=self.split_last)
        elif self.return_uv and not self.return_disp:
            self.imuv2uv = UNet(in_ch=6, out_ch=3)
        elif not self.return_uv and self.return_disp:
            self.imuv2disp = UNet(in_ch=6, out_ch=3)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        if self.return_uv and self.return_disp:
            # _, f_uvd = self.uvFeature(x)
            # y_uvd, _ = self.uvFeature2uvd(torch.cat((x, f_uvd), dim=1))
            # x4, x3, x2, x1, x0 = self.imuv2uvdisp_encoder(x)
            # y_uvd = self.imuv2uv_disp_decoder(x4, x3, x2, x1, x0)
            y_uvd, _ = self.imuv2uvdisp(x)
            output = {'uv': y_uvd[:, :3, :, :],
                      'disp': y_uvd[:, 3:, :, :]}
        elif self.return_uv and not self.return_disp:
            y_uv = self.imuv2uv(x)
            output = {'uv': y_uv}
        elif not self.return_uv and self.return_disp:
            y_disp = self.imuv2disp(x)
            output = {'disp': y_disp}
        return output

class DeepHumanUVNet(pl.LightningModule):
    def __init__(self, opt):
        super(DeepHumanUVNet, self).__init__()
        self.model = BaseModule(return_uv=opt.data.return_uv,
                                return_disp=opt.data.return_disp)
        self.automatic_optimization = True
        self.loss = LossBuilderHumanUV(opt=opt)
        self.learning_rate = 0.001  # opt.learning_rate
        self.log_every_t = 100  # opt.log_every_n_steps
        self.dr_loss = opt.data.dr_loss
        # self.use_amp = True
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def training_step(self, train_batch, batch_idx):
        # input = torch.cat((train_batch['image_cond'], train_batch['uv_cond']), dim=1)
        # opt = self.optimizers(self.automatic_optimization)
        # sch = self.lr_schedulers()
        # opt.zero_grad()
        # with torch.cuda.amp.autocast():
        #     pred_var = self.model(input)
        #     train_loss, log_dict = self.loss.forward(pred_var, train_batch)
        # self.scaler.scale(train_loss).backward(retain_graph=True)
        # self.scaler.step(opt)
        # self.scaler.update()
        # sch.step()

        input = torch.cat((train_batch['image_cond'], train_batch['uv_cond']), dim=1)
        # opt = self.optimizers(self.automatic_optimization)
        # sch = self.lr_schedulers()

        pred_var = self.model(input)
        train_loss, log_dict = self.loss.forward(pred_var, train_batch)

        # self.manual_backward(train_loss)
        # opt.step()

        # step at the last bach of each epoch.
        # if self.trainer.is_last_batch:
        #     sch.step()
        # opt.zero_grad()

        logs = {'train_loss': train_loss}
        if batch_idx % self.log_every_t == 0:
            log_dict['input'] = input[0, :3]
            log_dict['input_dense_uv'] = input[0, 3:6]
            input_color_grid = self.make_summary(log_dict)
            self.logger.experiment.add_scalar("Loss/Train", train_loss, self.global_step)
            self.logger.experiment.add_image("Images/Train", input_color_grid, self.global_step)
        return {'loss': train_loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        input = torch.cat((val_batch['image_cond'], val_batch['uv_cond']), dim=1)
        pred_var = self.model(input)
        val_loss, log_dict = self.loss.forward(pred_var, val_batch)
        return {'loss': val_loss}

    def test_step(self, val_batch, batch_idx):
        self.model.eval()
        if torch.is_tensor(val_batch):
            return self.model(val_batch, pred_uv=self.opt_uv)
        else:
            input = torch.cat((val_batch['image_cond'], val_batch['uv_cond']), dim=1)
            pred = self.model(input)
            test_loss, _ =  self.loss.forward(pred, val_batch)
            return {'loss': test_loss}

    @torch.no_grad()
    def in_the_wild_step(self, input):
        self.model.eval()
        return self.model(input)

    def make_summary(self, log_dict):
        log_list = []
        permute = [2, 1, 0]
        log_list.append(log_dict['input'][permute, :, :])
        log_list.append(log_dict['input_dense_uv'][permute, :, :].to(log_dict['input'].device))

        rot_num = len(log_dict['render_tgt_img'])
        num = 0
        if 'uv_pred' in log_dict:
            log_list.append(log_dict['uv_pred'][0][permute, :, :].to(log_dict['input'].device))

        if 'uv_tgt' in log_dict:
            log_list.append(log_dict['uv_tgt'][0][permute, :, :].to(log_dict['input'].device))

        if 'render_pred_img' in log_dict and 'render_tgt_img' in log_dict:
            for i in range(len(log_dict['render_tgt_img'])):
                log_list.append(log_dict['render_pred_img'][i][permute, :, :].to(log_dict['input'].device))
                log_list.append(log_dict['render_tgt_img'][i][0][permute, :, :].to(log_dict['input'].device))
                num += 1
                if num == rot_num:
                    num = 0
                    break

        if 'render_pred_normal' in log_dict and 'render_tgt_normal' in log_dict:
            for i in range(len(log_dict['render_pred_normal'])):
                log_list.append(log_dict['render_pred_normal'][i][permute, :, :].to(log_dict['input'].device))
                log_list.append(log_dict['render_tgt_normal'][i][0][permute, :, :].to(log_dict['input'].device))
                num += 1
                if num == rot_num:
                    num = 0
                    break

        input_color_grid = torchvision.utils.make_grid(log_list, normalize=True, scale_each=True, nrow=6)
        return input_color_grid


# def weight_init_basic(m):
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#     return m

def weight_init_basic(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

# for test
if __name__ == '__main__':
    # input = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    input = Variable(torch.randn(4, 2, 5, 5)).float().cuda()
    _, b = torch.Tensor.chunk(input, chunks=2, dim=1)
    print(b.shape)

    print(b)
    print(input)
