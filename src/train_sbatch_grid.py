import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse

import pytorch_lightning as pl
import torch_geometric.datasets
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import DataLoader

import wandb
from dataset import EgonetLoader, TUDataset
from model import *


def run_training_process_with_validation(run_params):
    print("######################################### NEW TRAIN on FOLD %d ######################################" % run_params.fold)

    if run_params.dataset == 'ENZYMES':
        dataset = torch_geometric.datasets.TUDataset('./data', run_params.dataset, hops=run_params.hops)
    else:
        dataset = TUDataset('./data', run_params.dataset, hops=run_params.hops)

    # dataset = TUDataset('./data',run_params.dataset)
    yy = [int(d.y) for d in dataset]
    fold = run_params.fold

    # Load or generate splits
    if not os.path.isfile('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds)):
        print('GENERATING %d FOLDS FOR %s' % (run_params.folds, run_params.dataset))
        skf = StratifiedKFold(n_splits=run_params.folds, random_state=1, shuffle=True)
        folds = list(skf.split(np.arange(len(yy)), yy))

        folds_split = []
        for fold in range(run_params.folds):
            train_i_split, val_i_split = train_test_split([int(i) for i in folds[fold][0]],
                                                          stratify=[n for n in np.asarray(yy)[folds[fold][0]]],
                                                          test_size=int(len(list(folds[fold][0])) * 0.1),
                                                          random_state=0)
            test_i_split = [int(i) for i in folds[fold][1]]
            folds_split.append([train_i_split, val_i_split, test_i_split])

        with open('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds), 'w') as f:
            f.write(json.dumps(folds_split))

    fold = run_params.fold
    with open('./data/folds/%s_folds_%d.txt' % (run_params.dataset, run_params.folds), 'r') as f:
        folds = json.loads(f.read())
    train_i_split, val_i_split, test_i_split = folds[fold]

    train_dataset = dataset[train_i_split]
    val_dataset = dataset[val_i_split]

    test_dataset = dataset[test_i_split]

    train_loader = EgonetLoader(train_dataset[:], batch_size=run_params.batch_size, shuffle=True)
    val_loader = EgonetLoader(val_dataset, batch_size=2000, shuffle=False)
    test_loader = EgonetLoader(test_dataset, batch_size=2000, shuffle=False)

    run_params.loss = torch.nn.CrossEntropyLoss()

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    run_params.in_features = train_dataset.num_features
    run_params.labels = train_dataset.num_features
    run_params.num_classes = train_dataset.num_classes

    model = Model(run_params)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=300,
        verbose=False,
        mode='min')

    if run_params.mask:
        run_params.max_epochs = run_params.max_epochs // 2

    wandb_logger = WandbLogger(
        name=f"graphK {run_params.date} fold {run_params.fold}",
        project=run_params.project,
        entity=run_params.group,
        offline=True
    )
    trainer = pl.Trainer.from_argparse_args(run_params, logger=wandb_logger,
                                            callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, datamodule=MyDataModule())
    if run_params.mask:
        trainer.max_epochs = trainer.max_epochs * 2
        # compute mask here
        print('==============> ', model.mask.shape)
        hidden = run_params.hidden
        data = DataLoader(train_dataset[:], batch_size=64, shuffle=True).__iter__().next()
        with torch.no_grad():
            b = torch.zeros(hidden)
            for tc in range(run_params.num_classes):
                with torch.no_grad():
                    output, x1 = model(data)
                    lo = torch.nn.functional.cross_entropy(output[data.y == tc, :], data.y[data.y == tc])

                    for j in range(hidden):
                        mask = torch.ones(hidden).float()
                        mask[j] = 0
                        model.mask.data = mask
                        output, x1 = model(data)
                        l = torch.nn.functional.cross_entropy(output[data.y == tc, :], data.y[data.y == tc])
                        b[j] = max(b[j], l - lo)
        model.mask.data = (b > b.max() * 0.1).float()
        print('==============> ', model.mask)

        print("Training after mask")
        trainer.fit(model, datamodule=MyDataModule())

    print("TRAINING FINISHED")
    print("################# TESTING #####################")
    trainer.test(datamodule=MyDataModule())
    print("################# VALIDATING #####################")
    trainer.validate(datamodule=MyDataModule())
    wandb.finish(0)
    if run_params.filename and wandb_logger.version:
        print(f'wandb id: {wandb_logger.version}')
        with open(run_params.filename, 'w') as f:
            f.write(f'{wandb_logger.version}\n{run_params.project}')


import json
import sys
from argparse import Namespace


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default='FullBackGL')
    parser.add_argument("--dataset", default='NCI1')
    parser.add_argument("--fold", default=0)
    parser.add_argument("--folds", default=10)

    parser.add_argument("--nodes", default=6, type=int)
    parser.add_argument("--labels", default=7, type=int)
    parser.add_argument("--hidden", default=16, type=int)

    parser.add_argument("--mlp_layers", default=2, type=int)
    parser.add_argument("--activation", default='relu')
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--hops", default=1, type=int)
    parser.add_argument("--kernel", default='gl', type=str)
    parser.add_argument("--normalize", default=True, type=bool)

    parser.add_argument("--pooling", default='add', type=str)

    parser.add_argument("--sparsity", default=0, type=float)
    parser.add_argument("--jsd_weight", default=0, type=float)
    parser.add_argument("--max_cc", default=True, type=bool)
    parser.add_argument("--mask", default=False, type=bool)

    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--lr_graph", default=1e-2, type=float)

    parser.add_argument("--optimize_masks", default=True)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--run", default=0)  # num filters
    parser.add_argument("--debug", default=False, type=lambda x: (str(x).lower() == 'true'))  # num filters
    parser.add_argument("--slurm", default=True, type=lambda x: (str(x).lower() == 'true'))  # num filters

    parser.add_argument("--filename", default='', type=str)
    return parser


if __name__ == "__main__":
    if sys.argv[1] == 'launch':
        params = Namespace(**json.loads(sys.argv[2]))
        parser = argparse.ArgumentParser()
        parser = pl.Trainer.add_argparse_args(parser)
        params = parser.parse_args(['--gpus', '0',
                                    '--log_every_n_steps', '20',
                                    '--progress_bar_refresh_rate', '10',
                                    '--check_val_every_n_epoch', '1'], namespace=params)
        run_training_process_with_validation(params)
