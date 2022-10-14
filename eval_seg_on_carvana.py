"""
Created on 11/06/21, by Simon Zhang
This is an evaluator for segmentation on Carvana. 
It inherits from the superclass `EvaluatorBase` in the ./base/eval.py and mainly writes overriding functions to focus on the task-specific evaluation.
"""

import argparse

import torch
import torch.nn.functional as F
from eval import EvaluatorBase
from utils.meter import AvgMeter
from loss.dice_loss import dice_coeff, multiclass_dice_coeff

class EvaluatorSeg(EvaluatorBase):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.dice_score = AvgMeter() # create the task-specific AvgMeter to track the dice score
    
    def test(self):
        return super().test()

    @torch.no_grad()
    def test_per_dataset(self):
        self.test_per_dataset_seg_on_carvana()

    def test_per_dataset_seg_on_carvana(self):
        self.dice_score.reset()
        self.model.eval()
        print(f"There are {self.opt.n_snap} samples in the dataset for test.")
        for iter, (batch, name) in enumerate(self.eval_loader):
            self.name = name[0]
            ##########################################
            # import ipdb; ipdb.set_trace()
            image, mask = batch['image'].to(device=self.device, dtype=torch.float32), batch['mask'].to(device=self.device, dtype=torch.long)
            mask_pred = self.model(image)
         
            mask = F.one_hot(mask, self.opt.n_classes).permute(0, 3, 1, 2).float()
            if self.opt.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                self.dice_score.update(dice_coeff(mask_pred, mask, reduce_batch_first=False))
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.opt.n_classes).permute(0, 3, 1, 2).float()
                self.dice_score.update(multiclass_dice_coeff(mask_pred[:, 1:, ...], mask[:, 1:, ...], reduce_batch_first=False))
            
            if self.opt.viz and (iter % self.opt.n_snap == 0):
                self.viz(image, 'origin')
                self.viz(mask, 'mask')
                self.viz(mask_pred, 'pred')

        print(f"Avg dice: {self.dice_score.avg}")
            
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--proj_name', type=str, default= "Test", help='Project name')
    parser.add_argument('--mode', type=str, default='direct', choices=['padding', 'direct'])
    parser.add_argument('--viz', type=bool, default=False, help="turn on/off the viz")
    parser.add_argument('--n_snap', type=int, default=50, help="viz per n_snap")

    parser.add_argument('--model_type', type=str, default= 'UNet', choices=['UNet', 'UNet++'], help='choose the model')
    parser.add_argument('--data_type', type=str, default= 'Carvana', choices=['Carvana'], help='data type')

    parser.add_argument('--data_dir', type=str, default='', help="test dateset path")
    parser.add_argument('--model_path', type=str, default='', required=True, help='path of saved model')
    parser.add_argument('--save_path', type=str, default='./test_viz/', help='save path')

    parser.add_argument('--train_val_sample_rate', type=float, default=0.08)
    parser.add_argument('--test_sample_rate', type=float, default=0.02)
    parser.add_argument('--patch_size', type=int, default=256)

    # network
    parser.add_argument('--n_channels', type=int, default=3, help="UNet input channel")
    parser.add_argument('--n_classes', type=int, default=2, help="UNet output classes")
    parser.add_argument('--n_layers', type=int, default=9)
    parser.add_argument('--n_feature', type=int, default=48)
    
    parser.add_argument('--pretrained', type=bool, default=True)

    opt, _ = parser.parse_known_args()

    evaluator = EvaluatorSeg(opt)
    evaluator.test()
