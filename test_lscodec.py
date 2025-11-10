import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



from dataloader import DataModule
from lscodec.models import loaders, LMGen
import os

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
if DEBUG_MODE:
    import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();

def main(args):

    pl.seed_everything(3407)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    

    if args.save_enhanced is not None:
        config['save_enhanced'] = args.save_enhanced
        Path(args.save_enhanced).parent.mkdir(parents=True, exist_ok=True)

    model = loaders.get_lscodec(filename=config['save_enhanced'], device=None, num_codebooks=16,config=config)

    model.eval()
        
    data_module = DataModule(**config['dataset_config'])
        
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        logger=False,
    )

    trainer.test(model, data_module, ckpt_path=config['ckpt_path'])




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    parser.add_argument('--recon_dir', type=str, default=None, help='Path to recon_dir')
    parser.add_argument('--save_enhanced', type=str, default=None, help='The dir path to save enhanced wavs.')
    args = parser.parse_args()
    main(args)