from __future__ import division

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.data_entry import select_train_loader
from model.model_entry import select_model
from utils.clipper import clip_gradient
from utils.earlystopper import EarlyStopping
from utils.logger import setup_logger
from utils.meter import AvgMeter
from utils.metrics import Metrics
from utils.optimizer import select_optimizer
from utils.scheduler import get_scheduler
from utils.seed import randomseed
from utils.timestamp import timestamp

class TrainerBase():
    def __init__(self, opt):
        # import ipdb; ipdb.set_trace()
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_loader, self.val_loader = select_train_loader(opt, is_train=True, is_val=True) # * split the dataset into training and validation dataset
        self.eval_loader = None # * implemented in another script ---> test.py

        self.model = select_model(opt).to(self.device) # * Model choice
        if opt.parallel: # Default: False
            self.model = torch.nn.DataParallel(self.model) # * Whether training in more than one epoch

        self.criterion = None # overridden by specific child trainer class
        self.optimizer = select_optimizer(opt, self.model) # * Optimizer choice
        self.scheduler = get_scheduler(self.optimizer, len(self.training_loader), opt) # * Learning rate scheduler
        self.logger = setup_logger(output=opt.output_dir, name=opt.proj_name) # * Event log setting
        self.writer = SummaryWriter(os.path.join(opt.output_dir, 'summary')) # * Tensorboard for visulization
        self.stopper = EarlyStopping() # * If not imporved for several epochs, then stop training
    
        self.train_step, self.val_step = 0, 0 # * Used as the Tensorboard log step
        self.min_val_loss = np.inf # * To check if save a checkpoint 
        self.train_loss_meter = AvgMeter() # * A record of the training loss, can output the overall average
        self.val_loss_meter = AvgMeter()
        self.metrics = Metrics() # * Metric calculation

        self.save_config() # * Save the configuration
   
    def train(self):
        for epoch in range(1, self.opt.n_epoch + 1):
            try:
                self.train_per_epoch(epoch)
                if epoch == self.opt.n_epoch:
                    self.save(epoch, status='last')
                # self.print_num_params() # * Change everytime? --> No
                if self.val_loader:
                    self.val_per_epoch(epoch)
                    if self.stopper.stop_flag:
                        self.logger.warning(f"Early stopping triggered: save model and exit. Bset score: {self.stopper.best_score}")
                        self.save(epoch, status='earlystop')
                        break
            except KeyboardInterrupt:
                self.logger.warning("Keyboard Interrupt: save model and exit.")
                os.makedirs(self.opt.output_dir, exist_ok=True)
                self.save(epoch, status='InT')
                break
        self.writer.close()


    def train_per_epoch(self, epoch): # ! This is actually a template for being overriding in the subclasses of TrainerBase.
        pass
    
    @torch.no_grad()
    def val_per_epoch(self, epoch):
        pass
    
    def train_per_epoch_template(self, epoch): 
        tic = time.time()
        self.train_loss_meter.reset()
        self.model.train()
        for iteration, (clean, _) in enumerate(self.training_loader): # verify this Checked. For ixi-T1, clean is of (N, H, W, 1), tensor, cpu.
            self.train_step += 1
            ########################################## Foward and loss calculation
            clean = clean.cuda()
            # import ipdb; ipdb.set_trace()
            input = None
            output = self.model(input)
            loss_all = None
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
    def val_per_epoch_template(self, epoch):
        self.val_loss_meter.reset()
        self.metrics.reset()
        self.model.eval()
        for iteration, (clean, _) in enumerate(self.val_loader):
            self.val_step += 1
            ########################################## # Forward calculation and loss
            # import ipdb; ipdb.set_trace()
            clean = clean.cuda()
            input = None
            output = self.model(input)
            loss_all = None
            ##########################################
            self.metrics.push(clean, output) # risky
            self.val_loss_meter.update(loss_all.item())
            self.writer.add_scalar('Val_loss_all_per_iter', loss_all.item(), global_step=self.val_step)
            if iteration == 0 or (iteration+1) % 500 == 0 or (iteration+1) == len(self.val_loader):
                self.logger.info(f"Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}], Iteration:[{iteration+1:05d}/{len(self.val_loader):05d}], Val_loss_all:{loss_all.item():.6f}")
        self.logger.info(f"Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}], Avg val loss:{self.val_loss_meter.avg}")
        self.logger.info(f"Avg PSNR: {self.metrics.psnr_meter.avg}, Avg SSIM: {self.metrics.ssim_meter.avg}")
        self.writer.add_scalar('Val_loss_all_per_epoch', self.val_loss_meter.avg, global_step=epoch)
        self.writer.add_scalar('Val_psnr_per_epoch', self.metrics.psnr_meter.avg, global_step=epoch)
        self.writer.add_scalar('Val_ssim_per_epoch', self.metrics.ssim_meter.avg, global_step=epoch)
        self.stopper(metrics=self.metrics.psnr_meter.avg, loss=False)
        if self.min_val_loss > self.val_loss_meter.avg:
            self.logger.info(f"Best val loss at Epoch:[{epoch:03d}/{self.opt.n_epoch:03d}].")
            self.min_val_loss = self.val_loss_meter.avg
            self.save(epoch, status='best')

    def save(self, epoch, status=''): # be careful when see the under-dash-line
        save_base_name = f"{self.opt.model_type}_{status}_ckpt.pth"
        save_path = os.path.join(self.opt.output_dir, save_base_name)
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Model saved in {save_path}.")

    def print_num_params(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.opt.model_type)
        print(self.model)
        self.logger.info(f"The number of parameters: {num_params:.3e}")
    
    def save_config(self):
        self.logger.info(f"Description: {self.opt.description}")
        self.logger.info(f"Model: [{self.opt.model_type}], Dataset: [{self.opt.data_type}], Lr_scheduler: [{self.opt.lr_scheduler}], optimizer: [{self.opt.optim}], batch size: [{self.opt.batch_size}]; All saved in {self.opt.output_dir}")
        with open(os.path.join(self.opt.output_dir, 'config_' + self.opt.proj_name + '.json'), 'w') as f:
            json.dump(vars(self.opt), f, indent=2)
        self.logger.info(f"Config saved to {os.path.join(self.opt.output_dir, 'config_' + self.opt.proj_name + '.json')}")
        self.logger.info(f"Current_device: {torch.cuda.get_device_name()}")
        
    def IsPrint(self, iteration, maxlength, gap=500): # Maybe modified later for better integration
        return iteration == 0 or (iteration + 1) % gap == 0 or (iteration+1) == maxlength

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--proj_name', type=str, required=True, help="Project name")
    parser.add_argument('--description', type=str, default= 'N/A', help="Description")
    parser.add_argument('--seed', type=int, default=None, help="Reproducibility")

    # type
    parser.add_argument('--model_type', type=str, required=True, choices=[""], help="Choose the model")
    parser.add_argument('--data_type', type=str, required=True, choices=[""], help="Choose the dataset") #

    # IO
    parser.add_argument('--data_dir', type=str, default='dataset', help="Dir to the dataset, this is only necessary for local dataset (FashionMNIST not included)") #
    parser.add_argument('--output_dir', type=str, required=True, help="Dir to save the results")
    parser.add_argument('--model_path', type=str, default='', help='Path of a pretraining model')

    parser.add_argument('--gpu_devices', type=str, default='0', choices=['0', '1', '0, 1'], help="To specify which GPU to use, there are 2 GPUs by default.")
    parser.add_argument('--parallel', type=bool, default=False, help="Parallel training flag")
    parser.add_argument('--pretrained', type=bool, default=False, help="Pretrained flag")

    # network
    # TODO
    
    # training
    parser.add_argument('--train_val_sample_rate', type=float, default=0.08, help="Only choose part of dataset used for training.")
    parser.add_argument('--test_sample_rate', type=float, default=0.02, help="Only choose part of dataset used for testing.")
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
    trainer = TrainerBase(opt)
    trainer.train()
