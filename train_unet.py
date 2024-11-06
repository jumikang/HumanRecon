import os
import glob
import hydra
import torch
import warnings
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.loader_unet import create_dataset
from models.unet.deep_human_models import DeepHumanUVNet
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(config_path="config", config_name="base_config_unet")
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
    logger = TensorBoardLogger(save_dir='logs/', name=f"{opt.exp}")
    trainer = pl.Trainer(
        devices=-1,
        precision='16',
        accelerator="auto",  # fsdp, ddp, auto
        callbacks=[checkpoint_callback],
        max_epochs=opt.train.epochs,
        check_val_every_n_epoch=opt.train.val_every_n_epoch,
        logger=logger,
        log_every_n_steps=opt.train.log_every_n_epoch,
        strategy="auto",
        num_sanity_val_steps=0
    )
    print('cuda_id: %d' % trainer.local_rank)
    model = DeepHumanUVNet(opt)
    trainset = create_dataset(opt.data, validation=False)
    validset = create_dataset(opt.data, validation=True)

    if opt.train.is_continue:
        checkpoint = sorted(glob.glob("checkpoints/epoch=*.ckpt"))[-1]
        trainer.fit(model, trainset, validset, ckpt_path=checkpoint)
    else:
        trainer.fit(model, trainset, validset)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn', force=True)
    # set "highest", "high", or "medium
    torch.set_float32_matmul_precision("medium")
    main()