import torch

def select_optimizer(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr / 10.0 * opt.batchsize, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError(f"There is no such optimizer called {opt.optim}")
    return optimizer