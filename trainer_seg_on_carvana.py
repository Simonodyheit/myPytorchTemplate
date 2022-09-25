"""
Created on 11/06/21, by Simon Zhang
This is a trainer for segmentation on the dataset Carvana. 
It inherits from the superclass `TrainerBase` in `./base/trainer.py` and mainly writes overriding functions to focus on the task-specific training.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch._C import device

from base.trainer import TrainerBase
from loss.dice_loss import dice_coeff, dice_loss, multiclass_dice_coeff
from utils.clipper import clip_gradient
from utils.meter import AvgMeter
from utils.seed import randomseed
from utils.timestamp import timestamp


class TrainerSeg(TrainerBase):
    def __init__(self, opt):
        super().__init__(opt) # for inheritance
        self.opt = opt
        self.dice_loss = dice_loss
        self.dice_score = AvgMeter()
        self.criterion_CE = torch.nn.CrossEntropyLoss()
        
    def train(self):
        return super().train()
    
    def train_per_epoch(self, epoch):
        self.train_per_epoch_seg_on_carvana(epoch)
    
    @torch.no_grad()
    def val_per_epoch(self, epoch):
        self.val_per_epoch_seg_on_carvana(epoch)

    def train_per_epoch_seg_on_carvana(self, epoch):
        tic = time.time()
        self.train_loss_meter.reset()
        self.model.train()
        for iteration, (batch, _) in enumerate(self.training_loader):
            self.train_step += 1
            ##########################################
            # import ipdb; ipdb.set_trace()
            images, masks = batch['image'].to(device=self.device, dtype=torch.float32), batch['mask'].to(device=self.device, dtype=torch.long)
            masks_pred = self.model(images)
            loss_all = self.criterion_CE(masks_pred, masks) \
                       + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   F.one_hot(masks, self.opt.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True)
            ##########################################
            self.optimizer.zero_grad()
            loss_all.backward()
            if self.opt.clip:
                clip_gradient(self.optimizer, self.opt)
            self.optimizer.step()
            if self.IsPrint(iteration, len(self.training_loader)):
                self.logger.info(f"Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}], Iteration:[{iteration+1:05d}/{len(self.training_loader):05d}], Train_loss_all:{loss_all.item():.6f}")
            self.train_loss_meter.update(loss_all.item(), n=self.opt.batch_size)
            self.writer.add_scalar('Train_loss_all_per_iter', loss_all.item(), global_step=self.train_step)
            self.scheduler.step()
        self.writer.add_scalar('Train_loss_all_per_epoch', self.train_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar('Lr_per_epoch', self.optimizer.param_groups[0]['lr'], global_step=epoch)
        self.logger.info(f"Epoch:{epoch}: Used time:{time.time()-tic:.2f}s; Lr:{self.optimizer.param_groups[0]['lr']:.2e}")
        self.logger.info(f"Epoch:{epoch}: Avg train loss:{self.train_loss_meter.avg}")

    @torch.no_grad()
    def val_per_epoch_seg_on_carvana(self, epoch):
        self.val_loss_meter.reset()
        self.metrics.reset()
        self.model.eval()
        self.dice_score.reset()
        for iteration, (batch, _) in enumerate(self.val_loader):
            self.val_step += 1
            ##########################################  # Foward and loss calculation
            # import ipdb; ipdb.set_trace()
            images, masks = batch['image'].to(device=self.device, dtype=torch.float32), batch['mask'].to(device=self.device, dtype=torch.long)
            masks_pred = self.model(images)
            loss_ce = self.criterion_CE(masks_pred, masks)
            loss_dice = self.dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(masks, self.opt.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
            loss_all = loss_ce + loss_dice
            ##########################################
            #! Score
            masks = F.one_hot(masks, self.opt.n_classes).permute(0, 3, 1, 2).float()
            if self.opt.n_classes == 1:
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                self.dice_score.update(dice_coeff(masks_pred, masks, reduce_batch_first=False))
            else:
                masks_pred = F.one_hot(masks_pred.argmax(dim=1), self.opt.n_classes).permute(0, 3, 1, 2).float()
                self.dice_score.update(multiclass_dice_coeff(masks_pred[:, 1:, ...], masks[:, 1:, ...], reduce_batch_first=False))
            ##########################################
            self.val_loss_meter.update(loss_all.item())
            self.writer.add_scalar('Val_loss_all_per_iter', loss_all.item(), global_step=self.val_step)
            if iteration == 0 or (iteration+1) % 500 == 0 or (iteration+1) == len(self.val_loader):
                self.logger.info(f"Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}], Iteration:[{iteration+1:05d}/{len(self.val_loader):05d}], Val_loss_all:{loss_all.item():.6f}")
        self.logger.info(f"Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}], Avg val loss:{self.val_loss_meter.avg}")
        self.logger.info(f"Avg dice: {self.dice_score.avg}")
        self.writer.add_scalar('Val_loss_all_per_epoch', self.val_loss_meter.avg, global_step=epoch)
        self.stopper(metrics=self.dice_score.avg, loss=False)
        if self.min_val_loss > self.val_loss_meter.avg:
            self.logger.info(f"Best val loss at Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}].")
            self.min_val_loss = self.val_loss_meter.avg
            self.save(epoch, status='best')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # ! Note that we can discard any of the following if we just want to keep the original setting in the base trainer.py file except for task-specific options like model_type and data_type
    # basic
    parser.add_argument('--proj_name', type=str, default="SegOnCarvana", help="Project name")
    parser.add_argument('--description', type=str, default= 'This is a trainer to demonstrate the template usage on segmentation on Carvana', help="Description")
    parser.add_argument('--seed', type=int, default=None, help="Reproducibility")

    # type
    # parser.add_argument('--model_type', type=str, required=True, choices=['UNet', 'UNet++'], help='Choose the model')
    # parser.add_argument('--data_type', type=str, required=True, choices=['Carvana'], help="Choose the dataset")
    parser.add_argument('--model_type', type=str, default = 'UNet', choices=['UNet', 'UNet++'], help='Choose the model')
    parser.add_argument('--data_type', type=str, default = 'Carvana', choices=['Carvana'], help="Choose the dataset")

    # IO
    parser.add_argument('--data_dir', type=str, required=True, help="Dir to the dataset, this is only necessary for local dataset (FashionMNIST not included)") #
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save the results")
    parser.add_argument('--model_path', type=str, default='', help='Path of a pre-training model')

    parser.add_argument('--gpu_devices', type=str, default='0', choices=['0', '1', '0, 1'], help="To specify which GPU to use, there are 2 GPUs by default.")
    parser.add_argument('--parallel', type=bool, default=False, help="Parallel training flag")
    parser.add_argument('--pretrained', type=bool, default=False, help="Pretrained flag")

    # network
    parser.add_argument('--n_channels', type=int, default=3, help="UNet input channel")
    parser.add_argument('--n_classes', type=int, default=2, help="UNet output classes")
    
    # training
    parser.add_argument('--train_val_sample_rate', type=float, default=0.08, help="Only choose part of dataset used for training for fast implementation. Need to be 1 when full dataset is desired!")
    parser.add_argument('--test_sample_rate', type=float, default=0.02, help="Only choose part of dataset used for testing for fast implementation. Need to be 1 when full dataset is desired!")
    parser.add_argument('--n_epoch', type=int, default=99)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--optim', type=str, default='adamW', choices=['adam', 'adamW', 'sdg'], help='choose the optimizer')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'], help='choose the lr scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[], nargs='+', help='for step scheduler, where to decay lr, can be a list')
    parser.add_argument('--lr_decay_steps', type=int, default=20, help='for step scheduler, step size to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='for step scheduler, decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--clip', type=float, default=0, help='gradient clipping margin')

    opt, _ = parser.parse_known_args()
    
    ts = timestamp()

    opt.output_dir = os.path.join(opt.output_dir, ts)
    
    opt.proj_name += '_' + ts
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

    if opt.seed:
        randomseed(opt)
    trainer = TrainerSeg(opt)
    trainer.train()
