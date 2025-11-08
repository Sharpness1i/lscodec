import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from natsort import natsorted
import subprocess
import os
from dataloader.data_module_parquet import CosyDataModule

from lscodec.models import loaders
import os
import torch
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

if DEBUG_MODE:
    import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();

class StepCheckpointCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step > 0 and global_step % 300000 == 0 and trainer.is_global_zero:
            ckpt_path = os.path.join(
                trainer.checkpoint_callback.dirpath,
                f"step-{global_step}.ckpt"
            )
            print(f" Saving checkpoint at {ckpt_path}")
            trainer.save_checkpoint(ckpt_path)


def save_as_safetensors(checkpoint, path):
    from safetensors.torch import save_file
    tensors = checkpoint["state_dict"]
    save_file(tensors, str(path).replace(".ckpt", ".safetensors"))

def main(args):
    pl.seed_everything(3407)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}' #change your folder, where to save files
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    config['ckpt_dir'] = ckpt_dir
    subprocess.run(['cp', args.config, ckpt_dir])
    subprocess.run(['cp', 'train.py', ckpt_dir])
    
    model = loaders.get_lscodec(filename=None, device=None,num_codebooks=16,config=config)
    
    model.train()
    
    model.teacher_feature_extractor.eval()  
    for param in model.teacher_feature_extractor.parameters():
        param.requires_grad = False

    
    strategy = 'ddp_find_unused_parameters_true' if config['devices'] > 1 else 'auto'
    
    data_module = CosyDataModule(args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step-{step}",
        every_n_train_steps=config['every_n_train_steps'], 
        save_top_k=-1,                
        save_last=False,
        save_weights_only=False,     
        monitor=None,                
        verbose=True,
    )
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        num_nodes=config['num_nodes'],
        devices=config['devices'],
        max_epochs=config['max_epochs'] if 'max_epochs' in config else None,
        max_steps=config['max_steps'] if 'max_steps' in config else None,
        val_check_interval=None,
        check_val_every_n_epoch=None,
        limit_val_batches=0,
        callbacks=[checkpoint_callback,StepCheckpointCallback()],
        logger=logger,
        strategy=strategy,
    )
    

    if config['resume_from_last_ckpt']:
        version_list = [str(p) for p in (Path(config['log_dir']) / 'ckpts').glob('*')]
        version_list = natsorted(version_list, reverse=True)
        ckpt_path = None
        for version in version_list:
            version = Path(version)
            ckpt_list = [str(p) for p in version.glob('*.ckpt')]
            ckpt_list = natsorted(ckpt_list, reverse=True)
            if len(ckpt_list) > 0:
                ckpt_path = ckpt_list[0]
                break
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, data_module, ckpt_path=config['resume'])



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='/root/code/lscodec/conf/config.yaml')
    parser.add_argument('--cosy_yaml', type=str, default='./conf/config.yaml')
    parser.add_argument('--uio_train_data', type=str, default='/primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list')

    args = parser.parse_args()
    main(args)