from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from data.Carvana import CarvanaDataset
from data.Imagenet_val import DataLoader_Imagenet_val

def get_dataset_by_type(args, is_train=False):
    if args.data_type == "Imagenet_val":
        dataset = DataLoader_Imagenet_val(args)
    elif args.data_type == "FashionMNIST":
        dataset = datasets.FashionMNIST(root=args.data_dir, train=is_train, download=True, transform=ToTensor()) 
    elif args.data_type == "Carvana":
        dataset = CarvanaDataset(images_dir=args.data_dir + '/train/', masks_dir=args.data_dir + '/train_masks/')
    else:
        raise TypeError(f"No such a data type as {args.data_type}!")
    return dataset

def select_train_loader(args, is_train=True, is_val=False):
    train_dataset = get_dataset_by_type(args, is_train)
    print('{} samples in training dataset overall (Dividable).'.format(len(train_dataset)))
    if args.train_val_sample_rate < 1:
        train_dataset = dataset_sample(train_dataset, args.train_val_sample_rate)
    if is_val:
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        if args.seed is not None:
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=Generator().manual_seed(args.seed))
        else:
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print(f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples over all.")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return train_loader, None

def select_eval_loader(args, is_train=False):
    eval_dataset = get_dataset_by_type(args, is_train)
    print('{} test samples overall.'.format(len(eval_dataset)))
    if args.test_sample_rate < 1:
        eval_dataset = dataset_sample(eval_dataset, args.test_sample_rate)
    eval_loader = DataLoader(eval_dataset, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return eval_loader

def dataset_sample(dataset, rate):
    partial_size = int(rate * len(dataset))
    print(f"Only {partial_size} are selected.")
    abandoned_size = len(dataset) - partial_size
    part_dataset, _ = random_split(dataset, [partial_size, abandoned_size])
    return part_dataset

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_type', type=str, default= '')
    parser.add_argument('--seed', type=int, default=None, help="Reproducibility")
    parser.add_argument('--train_val_sample_rate', type=int, default=0.08)
    parser.add_argument('--test_sample_rate', type=int, default=0.02)
    opt, _ = parser.parse_known_args()
    print(opt.seed)
    # # import ipdb; ipdb.set_trace()
    train_loader, val_loader = select_train_loader(opt, is_val=True)
    eval_loader = select_eval_loader(opt)
    print(len(train_loader))
    print(len(val_loader))
    print(len(eval_loader))
    print("Dataset loaded.")