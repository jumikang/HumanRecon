
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from models.diffusion.modules import FrozenCLIPImageEmbedder
# from transformers import CLIPTextModel, CLIPTokenizer


class DiffusionModel:
    def __init__(self, pretrained_model="runwayml/stable-diffusion-v1-5"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # read this from config file (and test recent models)
        self.pretrained_model = pretrained_model

        self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.pretrained_model,
                subfolder="scheduler")

        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #    self.pretrained_model,
        #    subfolder="tokenizer")

        # self.text_encoder = CLIPTextModel.from_pretrained(
        #    self.pretrained_model,
        #    subfolder="text_encoder").to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model,
            subfolder="vae").to(self.device)

        self.clip_encoder = FrozenCLIPImageEmbedder(
            device=self.device)

        # !only unet will be fine-tuned!
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model,
            subfolder="unet").to(self.device)

    def validate(self):
        pass

    def inference(self):
        pass

    def train(self, train_dataloader, params=None):
        self.vae.requires_grad_(False)
        self.clip_encoder.requires_grad_(False)
        # self.text_encoder.requires_grad_(False)
        self.unet.train()

        lr = float(params['lr'])
        epochs = params['epochs']
        log_every_t = params['log_every_t']
        print_every_t = params['print_every_t']

        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=lr,
            # betas=(args.adam_beta1, args.adam_beta2),
            # weight_decay=args.adam_weight_decay,
            # eps=args.adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=epochs * len(train_dataloader)
        )

        use_amp = False
        scaler = torch.amp.GradScaler(enabled=use_amp)
        # 'epsilon'  or 'v_prediction': predict noise or image
        prediction_type = 'epsilon'

        weight_dtype = torch.float16
        train_losses = []

        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(train_dataloader)):
                with torch.autocast(device_type=self.device, dtype=weight_dtype, enabled=use_amp):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["image_target"].to(self.device)  # , dtype=weight_dtype
                    ).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning, [0]: last_hidden_state
                    # token = self.tokenizer("a drawing of a green pokemon with red eyes")
                    # encoder_hidden_states = self.text_encoder(token["input_ids"].to(device))[0]
                    hidden_states_img = self.clip_encoder.encode(batch["image_cond"].to(self.device))
                    hidden_states_dense = self.clip_encoder.encode(batch["dense_cond"].to(self.device))

                    encoder_hidden_states = torch.concat((hidden_states_img, hidden_states_dense), dim=1)
                    if prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        self.noise_scheduler.register_to_config(prediction_type=prediction_type)
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                with torch.no_grad():
                    train_losses.append(loss.item())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

            # print loss
            avg_loss = sum(train_losses[-100:]) / 100
            print(f'Finished epoch {epoch + 1}. Average of the last 100 loss values: {avg_loss:05f}')