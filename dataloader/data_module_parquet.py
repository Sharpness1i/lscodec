import pytorch_lightning as pl
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.dataset.dataset import Dataset

class TrainDataLoadIter:
    def __init__(self, dataset_pipeline,steps_per_epoch):
        self.dataset = dataset_pipeline
        self.steps_per_epoch = steps_per_epoch
        
    def __iter__(self):
        for batch in self.dataset:
            yield batch

    def __len__(self):
        return self.steps_per_epoch


class CosyDataModule(pl.LightningDataModule):
    def __init__(self, args, gan=False, override_dict=None):
        super().__init__()
        self.args = args
        self.gan = gan
        self.override_dict = override_dict or {}
        with open(args.cosy_yaml, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=self.override_dict)

        data_pipeline = configs['data_pipeline_gan'] if gan else configs['data_pipeline']

        if gan and 'train_conf_gan' in configs:
            configs['train_conf'] = configs['train_conf_gan']
        elif 'train_conf' not in configs:
            configs['train_conf'] = {}

        configs['train_conf'].update(vars(args))
        self.configs = configs
        self.data_pipeline = data_pipeline

        self.train_dataset = Dataset(
            args.uio_train_data,
            data_pipeline=data_pipeline,
            mode='train',
            gan=gan,
            shuffle=True,
            partition=True
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = TrainDataLoadIter(self.train_dataset,steps_per_epoch=200000)

    def train_dataloader(self):
        return self.train_iter
