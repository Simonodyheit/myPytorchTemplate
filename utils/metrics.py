import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .meter import AvgMeter
from .func import normalize_zero_to_one
import torch

def check_sanity(im1, im2):
    assert type(im1) == type(im2), "Mismatched type for the two images."
    assert im1.shape == im2.shape, "Mismatched shape for the two images."

def check_batch_sanity(im1, im2):
    check_sanity(im1, im2)
    assert im1.ndim == 4 # ! What if it's more than 4 dimension?
    assert isinstance(im1, torch.Tensor), "Input batch should be a tensor."

def slice_psnr(gt, pred, maxval= None): # This is only for a single pair of images (gt, pred), especially in ndarray format. For a batch? Not for now, see the below.
    check_sanity(gt, pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    maxval = gt.max() if maxval is None else maxval
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def slice_ssim(gt, pred, maxval = None): # the in-built function has been already applied channel-wise, now we need focus on the batch-size, even it's just 1.
    check_sanity(gt, pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    color = True if pred.shape[1] == 3 else False
    maxval = gt.max() if maxval is None else maxval 
    return structural_similarity(gt, pred, data_range=maxval, multichannel=color)


def batch_psnr(gt, pred, maxval=None): # (N, 1, H, W) / (N, H, W, 1) ---> (H, W) Suitable only for gray images of ndim=4 (Tensor)
    check_batch_sanity(gt, pred)
    gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    psnr = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        psnr += peak_signal_noise_ratio(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val)
    return psnr / batch_size


def batch_ssim(gt, pred, maxval=None):
    color_flag = False
    check_batch_sanity(gt, pred)
    gt = gt.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)
    color_flag = True if pred.shape[-1] == 3 else False
    gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    batch_size = gt.shape[0]
    ssim = 0.0
    for i in range(batch_size):
        max_val = gt[i].max() if maxval is None else maxval
        ssim += structural_similarity(gt[i].squeeze(), pred[i].squeeze(), data_range=max_val, multichannel=color_flag)
    return ssim / batch_size


class Metrics:
    def __init__(self, batch_flag=True):
        self.psnr_meter = AvgMeter()
        self.ssim_meter = AvgMeter()
        if batch_flag:
            self.metric_funcs= {'PSNR':batch_psnr, 'SSIM':batch_ssim}
        else:
            self.metric_funcs= {'PSNR':slice_psnr, 'SSIM':slice_ssim}
        self.stats = {metric: None for metric in self.metric_funcs}

    def push(self, target, recons):
        for metric, func in self.metric_funcs.items():
            self.stats[metric]= func(target, recons)
        self.psnr_meter.update(self.stats['PSNR'])
        self.ssim_meter.update(self.stats['SSIM'])
    def reset(self):
        self.psnr_meter.reset()
        self.ssim_meter.reset()
        self.stats = {metric: None for metric in self.metric_funcs}

"""
def ssim(prediction, target):
    import cv2
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return psnr(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            psnrs = []
            for i in range(3):
                psnrs.append(psnr(img1[...,i], img2[..., i]))
            return np.array(psnrs).mean()
        elif img1.shape[2] == 1:
            return psnr(img1.squeeze(), img2.squeeze())
    else:
        raise ValueError('Wrong input image dimensions.')
"""

if __name__ == '__main__':
    import PIL.Image as Image
    target = Image.open("./asylum/metrics/101085_gray.png")
    noisy = Image.open("./asylum/metrics/101085_noisy.png")
    target_np = np.array(target)
    noisy_np = np.array(noisy)
    print('-' * 10)
    print(f"Shape: target: {target_np.shape}; noisy: {noisy_np.shape}")
    print(f"Max: target: {target_np.max()}; noisy: {noisy_np.max()}")
    print(f"Min: target: {target_np.min()}; noisy: {noisy_np.min()}")

    # print(f"PSNR: {calculate_psnr(target_np, noisy_np)}")
    # print(f"SSIM: {calculate_ssim(target_np, noisy_np)}")
    
    metric_slice = Metrics(batch_flag=False)
    metric_slice.push(target_np, noisy_np)
    print(metric_slice.psnr_meter.avg)
    print(metric_slice.ssim_meter.avg)

    target_np = np.expand_dims(target_np, 0)
    noisy_np = np.expand_dims(noisy_np, 0)
    target_np = np.expand_dims(target_np, -1)       
    noisy_np = np.expand_dims(noisy_np, -1)
    
    print('-'*10)
    print(f"Shape: target: {target_np.shape}; noisy: {noisy_np.shape}")
    print(f"Max: target: {target_np.max()}; noisy: {noisy_np.max()}")
    print(f"Min: target: {target_np.min()}; noisy: {noisy_np.min()}")

    metrics_batch = Metrics(batch_flag=True)
    metrics_batch.push(target_np, noisy_np)
    print(metrics_batch.psnr_meter.avg)
    print(metrics_batch.ssim_meter.avg)

    # target_np = normalize_zero_to_one(target_np)
    # noisy_np = normalize_zero_to_one(noisy_np)

    # print(f"Shape: target: {target_np.shape}; noisy: {noisy_np.shape}")
    # print(f"Max: target: {target_np.max()}; noisy: {noisy_np.max()}")
    # print(f"Min: targ:et: {target_np.min()}; noisy: {noisy_np.min()}")

    # # print(f"PSNR: {calculate_psnr(target_np, noisy_np)}")
    # # print(f"SSIM: {calculate_ssim(target_np, noisy_np)}")


    # metrics_batch = Metrics()
    # metrics_batch.push(target_np, noisy_np)
    # print(metrics_batch.psnr_meter.avg)
    # print(metrics_batch.ssim_meter.avg)
    # print('----------------------------------------')

    # target_np = np.clip(target_np * 255, 0, 255)
    # noisy_np = np.clip(noisy_np * 255, 0, 255)

    # print(f"Shape: target: {target_np.shape}; noisy: {noisy_np.shape}")
    # print(f"Max: target: {target_np.max()}; noisy: {noisy_np.max()}")
    # print(f"Min: target: {target_np.min()}; noisy: {noisy_np.min()}")

    # # print(f"PSNR: {calculate_psnr(target_np, noisy_np)}")
    # # print(f"SSIM: {calculate_ssim(target_np, noisy_np)}")

    # # target_np = np.expand_dims(target_np, -1)
    # # noisy_np = np.expand_dims(noisy_np, -1)

    # metrics_batch = Metrics()
    # metrics_batch.push(target_np, noisy_np)
    # print(metrics_batch.psnr_meter.avg)
    # print(metrics_batch.ssim_meter.avg)