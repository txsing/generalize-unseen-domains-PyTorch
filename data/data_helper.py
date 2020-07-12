from os.path import join, dirname

import numpy as np
from torchvision import datasets
import PIL.Image as Image
import pickle
import scipy.io
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, AdvDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset


vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = ["mnist", "mnist_m", "svhn", "usps", "syn"]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets + ["ALL"]


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


# source
# limit_source: True/False

# image_size
# min_scale, max_scale
# random_horiz_flip
# jitter
def get_train_dataloader(args):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer = get_train_transformers(args)
    limit = args.limit_source

    for dname in dataset_list:
        if dname in digits_datasets:
            return get_digital_train_dataloader(args, dname)
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
        train_dataset = JigsawDataset(name_train, labels_train, img_transformer=img_transformer)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(
            JigsawDataset(name_val, labels_val, img_transformer=get_val_transformer(args)
                )
            )
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def get_target_dataloaders(args):
    if args.target == 'ALL':
        loaders = []
        if args.source[0] in digits_datasets:
            target_domains = [item for item in digits_datasets if item != args.source[0] ]
            for dname in target_domains:
                loaders.append(get_digital_target_dataloader(args, dname))
            return loaders
        elif args.source[0] in pacs_datasets:
            target_domains = [item for item in pacs_datasets if item != args.source[0] ]
            for dname in target_domains:
                loaders.append(get_target_dataloader(args, dname))
            return loaders
    else:
        return [get_target_dataloader(args, args.target)]

def get_target_dataloader(args, dname):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % dname))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawDataset(names, labels, img_transformer=img_tr)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    print("Load %s, size: %d" % (dname, len(dataset)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader

# digit_dname = mnist/svhn
def get_digital_train_dataloader(args, dname):
    images, labels = None, []
    if dname == 'mnist':
        images, labels = load_mnist_dataset(split='train', flip_p=args.flip_p)

    dataset_digit = AdvDataset(images, labels)

    train_size, val_size = 0, 0
    if args.limit_source and len(dataset_digit) > args.limit_source:
        train_size = args.limit_source
    else:
        train_size = (int) (len(dataset_digit) * (1 - args.val_size))

    val_size = len(dataset_digit) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset_digit, [train_size , val_size])
    if val_size > (int)(train_size * args.val_size / (1 - args.val_size)):
        val_set = Subset(val_set, (int)(train_size * args.val_size / (1 - args.val_size)))

    train_set, val_set = ConcatDataset([train_set]), ConcatDataset([val_set])
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_digital_target_dataloader(args, dname):
    images, labels = None, []
    if dname == 'mnist':
        images, labels = load_mnist_dataset(split='test')
    elif dname == 'svhn':
        images, labels = load_svhn_dataset(split='test')
    elif dname == 'usps':
        images, labels = load_usps_dataset(split='test')
    elif dname == 'mnist_m':
        images, labels = load_mnist_m_dataset(split='test')
    elif dname == 'syn':
        images, labels = load_syn_dataset(split='test')

    dataset_digit = AdvDataset(images, labels)
    if args.limit_target and len(dataset_digit) > args.limit_target:
        dataset_digit = Subset(dataset_digit, args.limit_target)

    dataset = ConcatDataset([dataset_digit])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


# ref: https://github.com/ricvolpi/generalize-unseen-domains/blob/master/trainOps.py#load_mnist
def load_mnist_dataset(split='train', flip_p=0.1):
    print ('Loading MNIST dataset.')
    mnist = datasets.MNIST('~/Datasets/mnist', train=True, transform=None, 
                               target_transform=None, download=True)

    X = rgb32(mnist.train_data.numpy())
    X = torch.tensor(X).float()
    X = normalize(X)
    X = mnist_specific_augment(X, flip_p)
    return X, mnist.train_labels.int().tolist()

def mnist_specific_augment(X, flip_p):
    # 0.5-prob 1 or -1
    col_factor = torch.tensor( 
        np.random.binomial(1, 1 - flip_p, size=(len(X), 1, 1, 1)) * 2 - 1
    ).float()
    
    X = X * col_factor

    # col_factor = np.random.uniform(low=0.25, high=1.5, size=(len(X), 1, 1, 1))
    #X = (X * col_factor).astype(np.float32)
    
    # col_offset = np.random.uniform(low=-0.5, high=0.5, size=(len(X), 1, 1, 1))
    # X = (X + col_offset).astype(np.float32)
    return X

# ref: https://github.com/ricvolpi/generalize-unseen-domains/blob/master/trainOps.py#load_svhn
def load_svhn_dataset(split='test'):
    print ('Loading SVHN dataset.')
    svhn = datasets.SVHN('~/Datasets/svhn', split=split, transform=None,
                             target_transform=None, download=True)
    
    svhn_sample = torch.Tensor(svhn.data).float()
    svhn_label = torch.Tensor(svhn.labels).int()

    svhn_sample = normalize(svhn_sample)
    # svhn_sample = standardise_samples(svhn_sample)
    return svhn_sample, svhn_label.tolist()

def load_usps_dataset(split='train'):
    print ('Loading USPS dataset.')
    image_dir = '/home/users/hlli/Datasets/usps/usps_28x28.mat'
    usps = scipy.io.loadmat(image_dir)
    idx = 0 if split=='train' else 1
        
    X = usps['dataset'][idx][0].squeeze(1) * 255 
    # img_show(X,12)
    Y = usps['dataset'][idx][1].squeeze(1)
    
    X = rgb32(X)
    X = torch.tensor(X).float()
    X = normalize(X)
    return X, Y.tolist()

def load_syn_dataset(split='test'):
    print ('Loading SYN dataset.')
    image_dir = '/home/users/hlli/Datasets/syn/syn_number.mat'
    syn = scipy.io.loadmat(image_dir)
    
    X = syn[split+'_data']
    Y = syn[split+'_label'].squeeze(1)
    
    X = rgb32(X, resize=False)
    X = torch.tensor(X).float()
    X = normalize(X)
    return X, Y.tolist()

def load_mnist_m_dataset(split='test'):
    print ('Loading MNIST_M dataset.')
    image_dir = '/home/users/hlli/Datasets/mnist_m/mnistm_with_label.mat'
    mnist_m = scipy.io.loadmat(image_dir)
    
    X = mnist_m[split]
    Y = mnist_m['label_'+split].argmax(1)
    
    X = rgb32(X, resize=True)
    X = torch.tensor(X).float()
    X = normalize(X)
    return X, Y.tolist()

# adv_data => torch.Size([BatchSize, Channels, H, W])
# adv_labels => numpy 1-d int list  [1,2,3,4]
def append_adversarial_samples(args, data_loader, adv_data, adv_labels):
    datasets = data_loader.dataset.datasets

    dataset_adv = AdvDataset(adv_data, adv_labels)
    datasets.append(dataset_adv)

    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader

# image_size
# min_scale, max_scale
# random_horiz_flip
# jitter
def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(args.image_size, (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(img_tr)

# image_size
def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def rgb32(X, resize=True):
    X_32 = np.zeros((X.shape[0], 3, 32, 32), dtype=np.float32)
    for i in range(X.shape[0]):
        img = Image.fromarray(X[i])
        if resize:
            img = img.resize([32,32]) # 32 * 32 * 3
            img = img.convert('RGB')
        img = np.asarray(img, np.float32)
        img = img.transpose((2,0,1)) # 3 * 32 * 32
        X_32[i] = img.reshape(1,3,32,32)
    return X_32

def normalize(X):
    ### somehow [0,1] normalization will lead to slow training compared with [-1, 1]
    return (X - 128) / 128

def standardise_samples(X):
    X = X - X.mean(axis=(1,2,3), keepdims=True)
    X = X / X.std(axis=(1,2,3), keepdims=True)
    return X