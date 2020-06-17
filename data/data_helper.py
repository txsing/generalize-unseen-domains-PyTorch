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

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }


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

def get_target_dataloader(args):
    if args.target in digits_datasets:
        return get_digital_target_dataloader(args, args.target)
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawDataset(names, labels, img_transformer=img_tr)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader

# digit_dname = mnist/svhn
def get_digital_train_dataloader(args, dname):
    images, labels = None, []
    if dname == 'mnist':
        images, labels = load_mnist_dataset()
    elif dname == 'svhn':
        images, labels = load_svhn_dataset()

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

    dataset_digit = AdvDataset(images, labels)
    if args.limit_target and len(dataset_digit) > args.limit_target:
        dataset_digit = Subset(dataset_digit, args.limit_target)

    dataset = ConcatDataset([dataset_digit])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader

# ref: https://github.com/ricvolpi/generalize-unseen-domains/blob/master/trainOps.py#load_mnist
def load_mnist_dataset(split='train'):
    print ('Loading MNIST dataset.')
    mnist = datasets.MNIST('./data/datasets/mnist', train=True, transform=None, 
                               target_transform=None, download=True)

    MNIST_sample = torch.zeros(mnist.train_data.size(0), 3, 32, 32)
    for i in range(MNIST_sample.size(0)):
        img = Image.fromarray(mnist.train_data[i].numpy())
        img = img.resize([32,32])
        img = img.convert('RGB')
        img = np.asarray(img, np.float32)
        img = img.transpose((2,0,1))
        MNIST_sample[i] = torch.Tensor(img).unsqueeze(0)

# somehow [0,1] normalization will lead to slow training compared with [-1, 1]
#     MNIST_sample = MNIST_sample / 255
    MNIST_sample = (MNIST_sample - 128) / 128
    return MNIST_sample, mnist.train_labels.int().tolist()

#     image_file = 'train.pkl' if split=='train' else 'test.pkl'
#     image_dir = join('./data/datasets/', 'mnist', image_file)
#     with open(image_dir, 'rb') as f:
#         mnist = pickle.load(f, encoding='latin1')
#     images = mnist['X']
#     labels = mnist['y'].tolist()

#     images = images / 255.
#     images = np.stack((images,images,images), axis=1)  # grayscale to rgb
#     images = images.astype(np.float32)

#     return torch.from_numpy(np.squeeze(images)), labels

# ref: https://github.com/ricvolpi/generalize-unseen-domains/blob/master/trainOps.py#load_svhn
def load_svhn_dataset(split='train'):
    print ('Loading SVHN dataset.')
    svhn = datasets.SVHN('./data/datasets/svhn', split='train', transform=None, 
                             target_transform=None, download=True)
    
    svhn_sample = torch.Tensor(svhn.data).float()
    svhn_label = torch.Tensor(svhn.labels).int()
    
    
    # somehow [0,1] normalization will lead to slow training compared with [-1, 1]
#     svhn_sample = svhn_sample / 255
    svhn_sample = (svhn_sample - 128) / 128
    return svhn_sample, svhn_label.tolist()
#     image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'

#     image_dir = join('./data/datasets/', 'svhn', image_file)
#     svhn = scipy.io.loadmat(image_dir)
#     images = np.transpose(svhn['X'], [3, 2, 0, 1]) / 255.
#     images = images.astype(np.float32)

#     labels = svhn['y'].reshape(-1)
#     labels[np.where(labels==10)] = 0
#     return torch.from_numpy(images), labels.tolist()

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
