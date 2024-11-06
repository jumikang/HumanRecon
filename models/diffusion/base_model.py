
# import cv2
# import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from models.diffusion.modules import FrozenCLIPImageEmbedder


class BaseDiffusionModel(pl.LightningModule):
    def __init__(self, pretrained_model=None, params=None):
        super(BaseDiffusionModel, self).__init__()
        self.automatic_optimization = False
        # 'epsilon'  or 'v_prediction': predict noise or image
        self.pretrained_model = pretrained_model

        self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.pretrained_model,
                subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model,
            subfolder="vae")

        self.clip_encoder = FrozenCLIPImageEmbedder()

        # !only unet will be fine-tuned!
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model,
            subfolder="unet")

        # set parameters.
        if params is not None:
            self.prediction_type = params['prediction_type']
            self.log_every_t = params['log_every_t']
            self.log_every_t_val = params['log_every_t_val']
            self.lr = params.lr
        else:
            self.prediction_type = 'epsilon'
            self.log_every_t = 100
            self.log_every_t_val = 5
            self.lr = 1e-05

    def configure_optimizers(self):
        self.vae.requires_grad_(False)
        self.clip_encoder.requires_grad_(False)
        self.clip_encoder.eval()
        self.unet.train()

        optimizer = optim.Adam(self.unet.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # t: timesteps, cond: condition (for decoding images from noise in the logging step)
        noise_pred, noise_target, noisy_latent, t = self.shared_ddpm_step(train_batch, return_extra=True)
        train_loss = F.mse_loss(noise_pred.float(), noise_target.float(), reduction="mean")

        if self.automatic_optimization:
            self.backward(train_loss, retain_graph=True)
        else:
            self.manual_backward(train_loss)

        opt.step()
        if self.trainer.is_last_batch:
            sch.step()
        opt.zero_grad()

        logs = {'train_loss': train_loss}
        if batch_idx % self.log_every_t == 0:
            with torch.no_grad():
                denoised_latent = self.denoise(noisy_latent.detach(), noise_pred.detach(), t.detach())
                denoised_img = self.decode_image_from_latents(denoised_latent)
            self.save_logs(train_batch, train_loss, t=t.detach(), denoised_img=denoised_img, validation=False)
        return {'loss': train_loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        model_pred, target, noisy_latent, t = self.shared_ddpm_step(val_batch, return_extra=True, validation=True)
        val_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        logs = {'val_loss': val_loss}
        if batch_idx % self.log_every_t_val == 0:
            # denoise latent feature x_t at a single step
            with torch.no_grad():
                denoised_latent = self.denoise(noisy_latent.detach(), model_pred.detach(), t.detach())
                denoised_img = self.decode_image_from_latents(denoised_latent)
            self.save_logs(val_batch, val_loss, denoised_img=denoised_img, validation=True)
        return {'loss': val_loss, 'log': logs}

    @torch.no_grad()
    def one_step_denoise(self, wild_batch):
        """
        latent denoiser at a single step
        :param wild_batch: batch containing pred_img and condition images.
        :return: denoised texture map
        """
        model_pred, target, noisy_latent, t = self.shared_ddpm_step(wild_batch, return_extra=True, validation=True)
        with torch.no_grad():
            denoised_latent = self.denoise(noisy_latent.detach(), model_pred.detach(), t.detach())
            denoised_img = self.decode_image_from_latents(denoised_latent)
        return denoised_img

    @torch.no_grad()
    def every_step_denoise(self, wild_batch):
        pass

    def ddpm_from_noise(self, batch, return_extra=False):
        hidden_states_img = self.clip_encoder.encode(batch["image_cond"])
        hidden_states_dense = self.clip_encoder.encode(batch["dense_cond"])
        encoder_hidden_states = torch.concat((hidden_states_img, hidden_states_dense), dim=1)

        latents = torch.ones((1, 4, 64, 64)).to(self.device)
        noisy_latents = torch.randn_like(latents)
        bsz = noisy_latents.shape[0]
        timesteps = torch.ones((bsz,), device=noisy_latents.device) * self.noise_scheduler.config.num_train_timesteps - 1.0
        # timesteps = torch.ones((bsz,),
        #                        device=noisy_latents.device) * self.noise_scheduler.config.num_train_timesteps // 2

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        if return_extra:
            return model_pred, noisy_latents, timesteps.long()
        else:
            return model_pred

    def shared_ddpm_step(self, batch, validation=False, return_extra=False):
        # Convert images to latent space
        latents = self.vae.encode(
            batch["image_target"]
        ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)

        # Sample noise that we'll add to the latents
        bsz = noise.shape[0]

        # Sampling a timestep for each image
        if validation:
            # set timestep to t which means x_t ~ N(0, σ^2)
            timesteps = torch.ones((bsz,), device=latents.device) * self.noise_scheduler.config.num_train_timesteps - 1.0
        else:
            # set a random step in [0, t]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                      (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning, [0]: last_hidden_state
        hidden_states_img = self.clip_encoder.encode(batch["image_cond"])
        hidden_states_dense = self.clip_encoder.encode(batch["dense_cond"])
        encoder_hidden_states = torch.concat((hidden_states_img, hidden_states_dense), dim=1)

        if self.prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=self.prediction_type)
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        if return_extra:
            return model_pred, target, noisy_latents, timesteps
        else:
            return model_pred, target

    def denoise(self, noisy_samples, pred_noise, timesteps):
        # single step denoise function.
        # designed for almost noisy ...
        # for the subsequent add_noise calls
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=noisy_samples.device)
        timesteps = timesteps.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # reverse of adding noise.
        original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * pred_noise)/sqrt_alpha_prod
        # originally scaled by the scaling factor.
        return original_samples / self.vae.config.scaling_factor

    def decode_image_from_latents(self, latents):
        # !!!do not confuse vae.decoder and vae.decode!!!
        denoised_img = self.vae.decode(latents)
        denoised_img = torch.clip((denoised_img.sample.detach() + 1.0) / 2.0, min=0.0, max=1.0).to(latents.device)
        return denoised_img

    def save_logs(self, batch, loss, t=None, denoised_img=None, validation=False):
        log_dict = dict()
        # visualize the first image of a batch in a single row
        idx, nrows = 0, 3
        log_dict['target'] = batch["image_target"][idx]
        log_dict['dense_cond'] = batch["dense_cond"][idx]
        log_dict['img_cond'] = batch["image_cond"][idx]
        if denoised_img is not None:
            log_dict['denoised'] = denoised_img[idx]
            nrows += 1
        input_color_grid = self.make_summary(log_dict, nrows=nrows)
        mode = 'Train' if validation is False else 'Val'
        self.logger.experiment.add_scalar(f"Loss/{mode}", loss, self.global_step)
        if t is not None:
            self.logger.experiment.add_scalar(f"Timestep/{mode}", t[idx]/1000, self.global_step)
        self.logger.experiment.add_image(f"Images/{mode}", input_color_grid, self.global_step)

    @staticmethod
    def make_summary(log_dict, nrows=1):
        log_list = [log_dict['target'], log_dict['dense_cond'], log_dict['img_cond'], log_dict['denoised']]
        input_color_grid = torchvision.utils.make_grid(log_list, normalize=True, scale_each=True, nrow=nrows)
        return input_color_grid