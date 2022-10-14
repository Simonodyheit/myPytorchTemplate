from __future__ import division

import argparse
import os

import numpy as np
import torch
from data.data_entry import select_eval_loader
from model.model_entry import select_model
from PIL import Image
from utils.func import normalize_zero_to_one
from utils.metrics import Metrics

class EvaluatorBase():

    def __init__(self, opt):
        self.opt = opt
        # self.logger = setup_logger(output=opt.output_dir, name=opt.proj_name)
        self.eval_loader = select_eval_loader(opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = select_model(opt).cuda()
        self.model.eval()
        self.batch_metrics = Metrics()

    def test(self):
        if self.opt.viz:
            assert self.opt.save_path != '', "Save path should be valid."
            os.makedirs(os.path.join(self.opt.save_path, self.opt.model_path.split('/')[-2], self.opt.mode, self.opt.data_type), exist_ok=True)
        self.test_per_dataset() # ! modified here for quick test

    @torch.no_grad()
    def test_per_dataset(self):
        pass
    
    @torch.no_grad()
    def test_per_dataset_template(self):
        self.batch_metrics.reset()
        for iter, (clean, name) in enumerate(self.eval_loader):
            # import ipdb; ipdb.set_trace()
            self.name = name[0]
            clean.cuda()
            ####################################### Forward
            input = None
            pred = self.model(input)
            #######################################
            if self.opt.viz and (iter % self.opt.n_snap == 0):
                self.viz(clean, 'origin')
                self.viz(pred, 'pred')
            self.batch_metrics.push(clean, pred)

    def viz(self, data, postfix=''): # Only save tensor
        assert isinstance(data, torch.Tensor), "Input batch should be a tensor"
        N, C, *_ = data.shape
        assert data.ndim == 4 and N == 1, "The viz input shape should be (1, C, H, W)."

        save_path = os.path.join(self.opt.save_path, self.opt.model_path.split('/')[-2], self.opt.mode, self.opt.data_type, f"{self.name}_{postfix}.png")
        
        data = data[0].permute(1, 2, 0).detach().cpu().numpy() #  (1, C, H, W) ----> (H, W, C)
        
        data = normalize_zero_to_one(data) if data.max() >= 1.0 else data
        
        data = np.clip(data * 255 + 0.5, 0, 255)
        
        if C == 3:
            Image.fromarray(data.astype(np.uint8)).convert('RGB').save(save_path)
        
        else:
            Image.fromarray(data[..., 0].astype(np.uint8)).convert('L').save(save_path)
        
        print(f"Saved {self.name}_{postfix}.png in {os.path.join(self.opt.save_path, self.opt.data_type)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_name', type=str, default= "Test", help='Project name')
    parser.add_argument('--mode', type=str, default='direct', choices=['padding', 'direct'])
    parser.add_argument('--viz', type=bool, default=False, help="turn on/off the viz")
    parser.add_argument('--n_snap', type=int, default=1, help="viz per n_snap")

    parser.add_argument('--model_type', type=str, default= 'UNet', choices=['UNet'], help='choose the model')
    parser.add_argument("--noise_type", type=str, default="gauss25")
    parser.add_argument('--data_type', type=str, default= '', help='data type')

    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='', help="test dateset list")
    parser.add_argument('--model_path', type=str, default='', help='path of saved model')
    parser.add_argument('--save_path', type=str, default='./test_viz/', help='save path')

    parser.add_argument('--train_val_sample_rate', type=int, default=0.08)
    parser.add_argument('--test_sample_rate', type=int, default=0.02)
    parser.add_argument('--patch_size', type=int, default=256)

    parser.add_argument('--n_layers', type=int, default=9)
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=True)

    opt, _ = parser.parse_known_args()

    evaluator = EvaluatorBase(opt)
    evaluator.test()
    