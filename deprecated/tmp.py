
import os
import glob
import hydra
import torch
import warnings
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from lib.datasets import create_dataset
from deep_human_models import DeepHumanNet
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['HYDRA_FULL_ERROR'] = "1"


@hydra.main(config_path="confs", config_name="human_recon_base")
def main(opt):   
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())
    print(OmegaConf.to_yaml(opt))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        every_n_epochs=2,
        save_top_k=-1,
        save_last=True)
    # logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}")
    logger = TensorBoardLogger(save_dir='logs/', name=f"{opt.exp}")

    trainer = pl.Trainer(
        # gpus=-1,
        devices=-1,
        # accelerator="gpu",
        accelerator="auto",  # fsdp, ddp, auto
        callbacks=[checkpoint_callback],
        max_epochs=opt.max_epochs,
        check_val_every_n_epoch=opt.val_every_n_epoch,
        logger=logger,
        log_every_n_steps=opt.log_every_n_epoch,
        num_sanity_val_steps=0
    )

    model = DeepHumanNet(opt)
    trainset = create_dataset(opt.dataset.train)
    validset = create_dataset(opt.dataset.valid)

    if opt.model.is_continue == True:
        checkpoint = sorted(glob.glob("checkpoints/epoch=*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else: 
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    torch.cuda.empty_cache() 
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('medium')
    main()