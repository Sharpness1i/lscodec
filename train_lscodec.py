import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from natsort import natsorted
import subprocess
import os
import threading, queue, time, os
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file
from dataloader.data_module_parquet import CosyDataModule
# from dataloader.data_module_ori import DataModule
from dataloader.data_module_parquet_timeout import DataModule
from lscodec.models import loaders
import torch

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

if DEBUG_MODE:
    import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();





class AsyncSaver:
    """单例异步保存器（保证每次只跑一个后台save）"""
    _sem = threading.Semaphore(1)

    @staticmethod
    def save_async(obj, path):
        def _save_thread(data, file_path):
            try:
                torch.save(data, file_path)
            finally:
                AsyncSaver._sem.release()

        AsyncSaver._sem.acquire()
        t = threading.Thread(target=_save_thread, args=(obj, path), daemon=True)
        t.start()



class NonBlockingModelCheckpoint_ckpt(pl.Callback):
    def __init__(self, dirpath, every_n_train_steps: int, filename_tpl="step-{step:06d}"):
        super().__init__()
        self.dirpath = dirpath
        self.every = every_n_train_steps
        self.filename_tpl = filename_tpl
        os.makedirs(self.dirpath, exist_ok=True)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every > 0 and step > 0 and (step % self.every == 0):

            # ---- 1) CPU clone ----
            with torch.no_grad():
                model_sd = {k: v.detach().cpu() for k, v in pl_module.state_dict().items()}

            # ---- 2) collect optimizer / scheduler ----
            opt_sd = [opt.state_dict() for opt in trainer.optimizers]

            scheduler_states = []
            for cfg in trainer.strategy.lr_scheduler_configs:
                sch = cfg.scheduler
                if hasattr(sch, "state_dict"):
                    scheduler_states.append(sch.state_dict())

            ckpt = {
                "state_dict": model_sd,
                "optimizer_states": opt_sd,
                "lr_schedulers": scheduler_states,
                "global_step": step,
                "epoch": trainer.current_epoch,
            }

            fname = self.filename_tpl.format(step=step) + ".ckpt"
            fpath = os.path.join(self.dirpath, fname)

            print(f"[Async-CKPT] enqueue save: {fpath}")
            AsyncSaver.save_async(ckpt, fpath)



def main(args):
    pl.seed_everything(3407)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}' #change your folder, where to save files
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    config['ckpt_dir'] = ckpt_dir
    subprocess.run(['cp', args.config, ckpt_dir])
    
    model = loaders.get_lscodec(filename=None, device=None,num_codebooks=16,config=config)
    model.train()
    if config.get('use_distill'):
        model.teacher_feature_extractor.eval()  
        for param in model.teacher_feature_extractor.parameters():
            param.requires_grad = False

    # data_module = CosyDataModule(args)
     
    data_module = DataModule(**config['dataset_config'])
    
    data_module.train_kwargs['samples_per_epoch'] = args.samples_per_epoch
    
    
    # 命令行传参覆盖yaml
    if args.batch_size is not None:
        config['dataset_config']['train_kwargs']['batch_size'] = args.batch_size
    if args.devices is not None:
        config['devices'] = args.devices
    if args.num_nodes is not None:
        config['num_nodes'] = args.num_nodes

    strategy = 'ddp_find_unused_parameters_true' if config['devices'] > 1 else 'auto'
     
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename="step-{step:06d}",
    #     every_n_train_steps=save_steps, 
    #     save_top_k=-1,                
    #     save_last=False,
    #     save_weights_only=False,     
    #     monitor=None,                
    #     verbose=True,
    # )
    
    
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    non_block_callbacks = [
        NonBlockingModelCheckpoint_ckpt(
            dirpath=str(ckpt_dir),
            every_n_train_steps=config['every_n_train_steps'],
            filename_tpl="step-{step:06d}",
        )
    ]
    
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        num_nodes=config['num_nodes'],
        devices=config['devices'],
        max_epochs=config['max_epochs'] if 'max_epochs' in config else None,
        max_steps=config['max_steps'] if 'max_steps' in config else None,
        check_val_every_n_epoch=None,
        limit_val_batches=0,
        #callbacks=[ checkpoint_callback,StepCheckpointCallback()],
        callbacks=non_block_callbacks,
        logger=logger,
        strategy=strategy,
    )
    trainer.save_ckpt_step = args.save_ckpt_step
    
        # ----- resume from *.safetensors -----
    ckpt_path = None
    if config['resume_from_last_ckpt']:
        version_dirs = natsorted((Path(config['log_dir']) / "ckpts").glob("*"), reverse=True)
        for v in version_dirs:
            # 先找 safetensors
            ckpts = list(v.glob("*.safetensors"))
            if not ckpts:
                # 再找老的 ckpt (兼容)
                ckpts = list(v.glob("*.ckpt"))
            if ckpts:
                ckpt_path = natsorted(ckpts, reverse=True)[0]
                break

    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model._resume_state = state   # 暂存，等 on_fit_start 恢复 optimizer/scheduler

        trainer.fit(model, data_module)  # ❗ 不传 ckpt_path

    else:
        trainer.fit(model, data_module, ckpt_path=config['resume'])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='/root/code/lscodec/conf/config.yaml')
    parser.add_argument('--cosy_yaml', type=str, default='./conf/config.yaml')
    parser.add_argument('--uio_train_data', type=str, default='/primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list')
    parser.add_argument('--samples_per_epoch', type=int, default=1200000)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes', type=int, default=None)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--save_ckpt_step', type=int, default=1000)
    
    
    args = parser.parse_args()
    main(args)