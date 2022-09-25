import torch
from model.vanilla_model import Vanilla
from model.UNet import UNet
from model.UNetpp import NestedUNet


def select_model(opt):
    type2model = {
        'Vanilla': Vanilla() if opt.model_type == 'Vanilla' else None,
        'UNet' : UNet(n_channels=opt.n_channels, n_classes=opt.n_classes, bilinear=True) if opt.model_type == "UNet" else None,
        'UNet++' : NestedUNet(in_ch=opt.n_channels, out_ch=opt.n_classes) if opt.model_type == "UNet++" else None
    }
    model = type2model[opt.model_type]
    if opt.pretrained:
            if opt.model_type != 'resnet50':
                assert '.' in opt.model_path, "The model_path points to no file." # rough assertion
                model = load_match_dict(model, opt)
            print("Pretrained model loaded.")
    return model

def load_match_dict(model, opt):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(opt.model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                    k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":

    import numpy as np
    import argparse
    import time
    import torch
    import os
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--model_type', type=str, default= 'UNet', help='choose the model')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--pretrained', type=bool,default=False)
    parser.add_argument('--patch_size', type=int, default=32)
    opt, _ = parser.parse_known_args()

    x = torch.rand(10, 1, opt.patch_size, opt.patch_size)
    print(x.shape)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    print(f"Current_device: {torch.cuda.get_device_name()}")
    tic = time.time()
    x = x.cuda()
    print(f"Used time in x.cuda: {time.time() - tic}")
    net = select_model(opt)
    tic = time.time()
    net.cuda()
    print(f"Used time in model.cuda: {time.time() - tic}")
    tic = time.time()
    y = net(x)
    print(f"Used time in inference: {time.time() - tic}")
    print(y.shape)