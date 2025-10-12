import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from natsort import natsorted
import subprocess
from model import Model
from dataloader import DataModule
import os
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
if DEBUG_MODE:
    import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();
def main(args):
    pl.seed_everything(3407)    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}' #change your folder, where to save files
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config['ckpt_dir'] = ckpt_dir
    subprocess.run(['cp', 'conf/config.yaml', ckpt_dir])
    
    subprocess.run(['cp', 'train.py', ckpt_dir])
    # from moshi.models import loaders, LMGen
    model = Model(config=config)
    
    data_module = DataModule(**config['dataset_config'])
    checkpoint_callback_last = ModelCheckpoint(dirpath=ckpt_dir, save_on_train_epoch_end=True, filename='{epoch}-last')
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        num_nodes=config['num_nodes'],
        devices=config['devices'],
        num_sanity_val_steps=0,
        max_epochs=config['max_epochs'] if 'max_epochs' in config else None,
        max_steps=config['max_steps'] if 'max_steps' in config else None,
        val_check_interval=config['val_check_interval'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in config else 1,
        limit_val_batches=config['limit_val_batches'] if 'limit_val_batches' in config else None,
        #gradient_clip_val=config['gradient_clip_val'] if model.automatic_optimization else None,
        callbacks=[checkpoint_callback_last],
        logger=logger,
        strategy="auto" if len(config['devices']) == 1 else 'ddp_find_unused_parameters_true',
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
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    args = parser.parse_args()
    main(args)