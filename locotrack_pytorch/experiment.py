import os
import configparser
import argparse
import logging
from functools import partial
from typing import Any, Dict, Optional, Union

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model_utils import restore_model_from_jax_checkpoint

from data.kubric_data import KubricData
from models.locotrack_model import LocoTrack
import model_utils
from data.evaluation_datasets import get_eval_dataset
from data.human_data import VideoTrackingDataset


class LocoTrackModel(L.LightningModule):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_forward_kwargs: Optional[Dict[str, Any]] = None,
        loss_name: Optional[str] = 'tapir_loss',
        loss_kwargs: Optional[Dict[str, Any]] = None,
        query_first: Optional[bool] = False,
        optimizer_name: Optional[str] = 'Adam',
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_name: Optional[str] = 'OneCycleLR',
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        model_name = model_kwargs.pop('model_name', 'tapnext')
        self.model_name = model_name
        if model_name == 'locotrack':
            self.model = LocoTrack(**(model_kwargs or {}))
            # ckpt_path = 'locotrack_base.ckpt'
            # locotrack_weight = torch.load(ckpt_path)['state_dict']
            # self.model.load_state_dict(locotrack_weight)
        elif model_name == 'tapnet':
            from models.tapnet import TAPNet
            weights_path = (model_kwargs or {}).pop('weights_path', None)
            self.model = TAPNet(**(model_kwargs or {}))
            if weights_path is not None:
                tapnet_weight = torch.load(weights_path)
                self.model.load_state_dict(tapnet_weight)
        elif model_name == 'tapnext':
            from models.tapnext import TAPNext
            weights_path = (model_kwargs or {}).pop('weights_path', None)
            self.model = TAPNext((256, 256), **(model_kwargs or {}))
            if weights_path is not None:
                restore_model_from_jax_checkpoint(self.model, weights_path)
            
        self.model_forward_kwargs = model_forward_kwargs or {}
        self.loss = partial(model_utils.__dict__[loss_name], **(loss_kwargs or {}))
        self.query_first = query_first

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 2e-3}
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {'max_lr': 2e-3, 'pct_start': 0.05, 'total_steps': 300000}

    def training_step(self, batch, batch_idx):
        select_data = int(torch.rand(1).item() < 0.5)
        # select_data = 0
        batch = batch[select_data]
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(
            batch, 
            output, 
            huber_loss_margin=0.0,
            prob_loss_weight=0.0 if select_data == 1 else 1.0,
            occ_loss_weight=0.0 if select_data == 1 else 1.0,
            exclude_predicted_occlusion=False,
            query_points=batch['query_points'],
            use_smooth_deadzone_loss=(select_data == 1), # Use smooth deadzone loss for human data
            is_human=(select_data == 1),
        )
        
        self.log_dict(
            {f'train/{k}': v.item() for k, v in loss_scalars.items()},
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)
        
        log_prefix = 'val/'
        if dataloader_idx is not None:
            log_prefix = f'val/data_{dataloader_idx}/'

        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")

    def _tapnext_online_forward(self, video, query_points):
        """Run TAPNext frame-by-frame using online tracking state to avoid OOM on long videos."""
        B, T, H, W, C = video.shape
        all_tracks = []
        all_track_logits = []
        all_visible_logits = []

        # First frame: initialize tracking state
        output = self.model(video[:, :1], query_points, **self.model_forward_kwargs)
        state = output['tracking_state']
        all_tracks.append(output['tracks'])
        all_track_logits.append(output['track_logits'])
        all_visible_logits.append(output['visible_logits'])

        # Subsequent frames: pass state forward
        for k in range(1, T):
            output = self.model(video[:, k:k+1], state=state, **self.model_forward_kwargs)
            state = output['tracking_state']
            all_tracks.append(output['tracks'])
            all_track_logits.append(output['track_logits'])
            all_visible_logits.append(output['visible_logits'])

        tracks = torch.cat(all_tracks, dim=2)           # (B, Q, T, 2)
        track_logits = torch.cat(all_track_logits, dim=1)  # (B, T, Q, 512)
        visible_logits = torch.cat(all_visible_logits, dim=1)  # (B, T, Q, 1)

        return {
            'tracks': tracks,
            'track_logits': track_logits,
            'occlusion': -visible_logits.squeeze(-1).permute(0, 2, 1),
            'visible_logits': visible_logits,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if self.model_name == 'tapnext':
            output = self._tapnext_online_forward(batch['video'], batch['query_points'])
        else:
            output = self.model(batch['video'], batch['query_points'], **self.model_forward_kwargs)
        if self.model_name == 'tapnext' and not self.query_first:
            T = batch['video'].shape[1]
            query_points_rev = batch['query_points'].clone()
            query_points_rev[..., 0] = T - 1 - query_points_rev[..., 0]
            output_reverse = self._tapnext_online_forward(
                torch.flip(batch['video'], dims=[1]),
                query_points_rev,
            )

            reverse_tracks = torch.flip(output_reverse['tracks'], dims=[2])
            reverse_track_logits = torch.flip(output_reverse['track_logits'], dims=[1])
            reverse_visible_logits = torch.flip(output_reverse['visible_logits'], dims=[1])

            time_idx = torch.arange(
                output['tracks'].shape[2],
                device=output['tracks'].device,
                dtype=batch['query_points'].dtype,
            ).view(1, 1, -1)
            forward_mask = time_idx >= batch['query_points'][..., 0:1]
            forward_mask_tracks = forward_mask.unsqueeze(-1)
            forward_mask_logits = forward_mask.transpose(1, 2).unsqueeze(-1)

            merged_tracks = torch.where(
                forward_mask_tracks,
                output['tracks'],
                reverse_tracks,
            )
            merged_track_logits = torch.where(
                forward_mask_logits,
                output['track_logits'],
                reverse_track_logits,
            )
            merged_visible_logits = torch.where(
                forward_mask_logits,
                output['visible_logits'],
                reverse_visible_logits,
            )

            output = {
                'tracks': merged_tracks,
                'track_logits': merged_track_logits,
                'visible_logits': merged_visible_logits,
                'occlusion': -merged_visible_logits.squeeze(-1).permute(0, 2, 1),
            }
        loss, loss_scalars = self.loss(batch, output)
        metrics = model_utils.eval_batch(batch, output, query_first=self.query_first)

        log_prefix = 'test/'
        if dataloader_idx is not None:
            log_prefix = f'test/data_{dataloader_idx}/'
        
        self.log_dict(
            {log_prefix + k: v for k, v in loss_scalars.items()},
            logger=True,
            sync_dist=True,
        )
        self.log_dict(
            {log_prefix + k: v.item() for k, v in metrics.items()},
            logger=True,
            sync_dist=True,
        )
        logging.info(f"Batch {batch_idx}: {metrics}")
        
    def configure_optimizers(self):
        weights = [p for n, p in self.named_parameters() if 'bias' not in n]
        bias = [p for n, p in self.named_parameters() if 'bias' in n]

        optimizer = torch.optim.__dict__[self.optimizer_name](
            [
                {'params': weights, **self.optimizer_kwargs},
                {'params': bias, **self.optimizer_kwargs, 'weight_decay': 0.}
            ]
        )
        scheduler = torch.optim.lr_scheduler.__dict__[self.scheduler_name](optimizer, **self.scheduler_kwargs)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(
    mode: str,
    save_path: str,
    val_dataset_path: str,
    ckpt_path: str = None,
    kubric_dir: str = '',
    human_data_dir: str = '',
    precision: str = '32',
    batch_size: int = 1,
    val_check_interval: Union[int, float] = 5000,
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 1.0,
    max_steps: int = 300_000,
    model_kwargs: Optional[Dict[str, Any]] = None,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
    loss_name: str = 'tapir_loss',
    loss_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_name: str = 'Adam',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    scheduler_name: str = 'OneCycleLR',
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    # query_first: bool = False,
):
    """Train the LocoTrack model with specified configurations."""
    seed_everything(42, workers=True)

    model = LocoTrackModel(
        model_kwargs=model_kwargs,
        model_forward_kwargs=model_forward_kwargs,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        query_first='q_first' in mode,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_name=scheduler_name,
        scheduler_kwargs=scheduler_kwargs,
    )
    # model.load_state_dict(torch.load('locotrack_base.ckpt')['state_dict'], strict=True)
    if ckpt_path is not None and 'train' in mode:
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    logger = WandbLogger(project='AnthroTAP', save_dir=save_path, id=os.path.basename(save_path))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_last=True,
        save_top_k=3,
        mode="max",
        monitor="val/average_pts_within_thresh",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
    )

    eval_dataset = get_eval_dataset(
        mode=mode,
        path=val_dataset_path,
    )
    eval_dataloder = {
        k: DataLoader(
            v,
            batch_size=1,
            shuffle=False,
        ) for k, v in eval_dataset.items()
    }

    if 'train' in mode:
        trainer = L.Trainer(
            strategy=DDPStrategy(find_unused_parameters=True, broadcast_buffers=False),
            logger=logger,
            precision=precision,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            max_steps=max_steps,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
        )
        train_dataloader_kubric = KubricData(
            global_rank=trainer.global_rank, 
            world_size=trainer.world_size,
            data_dir=kubric_dir, 
            batch_size=batch_size * trainer.world_size,
        )

        dataset_human = VideoTrackingDataset(
            folder_list=[human_data_dir],
            sequence_length=48,
            dilation=2,
            output_size=(256, 256),
            num_queries=256,
            min_visible_frames=4,
            augment=True,
            augment_geom=True,
            aug_hflip_prob=0.5,
            aug_color_jitter=None,
            video_base_dir=human_data_dir,
        )
        train_dataloader_human = DataLoader(
            dataset_human,
            batch_size=batch_size,
            shuffle=True,
            num_workers=trainer.world_size * 4,
        )
        train_dataloader = [train_dataloader_kubric, train_dataloader_human]
        # train_dataloader = [train_dataloader_human]

        trainer.fit(model, train_dataloader, eval_dataloder, ckpt_path=ckpt_path)
    elif 'eval' in mode:
        trainer = L.Trainer(strategy='ddp', logger=logger, precision=precision)
        trainer.test(model, eval_dataloder, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate the LocoTrack model.")
    parser.add_argument('--config', type=str, default='config.ini', help="Path to the configuration file.")
    parser.add_argument('--mode', type=str, required=True, help="Mode to run: 'train' or 'eval' with optional 'q_first' and the name of evaluation dataset.")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument('--save_path', type=str, default='snapshots', help="Path to save the logs and checkpoints.")
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # Extract parameters from the config file
    train_params = {
        'mode': args.mode,
        'ckpt_path': args.ckpt_path,
        'save_path': args.save_path,
        'val_dataset_path': eval(config.get('TRAINING', 'val_dataset_path', fallback='{}')),
        'kubric_dir': config.get('TRAINING', 'kubric_dir', fallback=''),
        'human_data_dir': config.get('TRAINING', 'human_data_dir', fallback=''),
        'precision': config.get('TRAINING', 'precision', fallback='32'),
        'batch_size': config.getint('TRAINING', 'batch_size', fallback=1),
        'val_check_interval': config.getfloat('TRAINING', 'val_check_interval', fallback=5000),
        'log_every_n_steps': config.getint('TRAINING', 'log_every_n_steps', fallback=10),
        'gradient_clip_val': config.getfloat('TRAINING', 'gradient_clip_val', fallback=1.0),
        'max_steps': config.getint('TRAINING', 'max_steps', fallback=300000),
        'model_kwargs': eval(config.get('MODEL', 'model_kwargs', fallback='{}')),
        'model_forward_kwargs': eval(config.get('MODEL', 'model_forward_kwargs', fallback='{}')),
        'loss_name': config.get('LOSS', 'loss_name', fallback='tapir_loss'),
        'loss_kwargs': eval(config.get('LOSS', 'loss_kwargs', fallback='{}')),
        'optimizer_name': config.get('OPTIMIZER', 'optimizer_name', fallback='Adam'),
        'optimizer_kwargs': eval(config.get('OPTIMIZER', 'optimizer_kwargs', fallback='{"lr": 2e-3}')),
        'scheduler_name': config.get('SCHEDULER', 'scheduler_name', fallback='OneCycleLR'),
        'scheduler_kwargs': eval(config.get('SCHEDULER', 'scheduler_kwargs', fallback='{"max_lr": 2e-3, "pct_start": 0.05, "total_steps": 300000}')),
    }

    train(**train_params)
