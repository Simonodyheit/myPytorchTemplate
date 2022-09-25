def clip_gradient(optimizer, opt):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-opt.clip, opt.clip)