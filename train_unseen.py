import argparse

import os

import torch
from torch import optim
#from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets, pacs_datasets, digits_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
import numpy as np
import random


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Unseen training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU index")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data a stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")

    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate for classifiction training")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")

    parser.add_argument("--K", type=int, default=6, help="Number of whole adversarial phases")
    parser.add_argument("--T_min", type=int, default=10, help="Number of iterations in Min-phase")
    parser.add_argument("--T_max", type=int, default=15, help="Number of iterations in Max-phase")
    parser.add_argument("--gamma", type=float, default=1.0, help="Higher value leads to stricter distance constraint")
    parser.add_argument("--adv_learning_rate", type=float, default=1.0, help="Learning rate for adversarial training")
    parser.add_argument("--flip_p", type=float, default=0.1, help="flip probability")

    # nesterov 是一种梯度下降的方法
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")
    parser.add_argument("--adam", action='store_true', help="Use adam for classification training")
    parser.add_argument("--decay", type=float, default=1.0, help="learning rate decay for min-phase")

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        print('config: mnist %g flip to %s!' % (args.flip_p, args.target) )
        self.device = device

        model = model_factory.get_network(args.network)(classes=args.n_classes)
        self.model = model.to(device)

        # The training dataset get divided into two parts (Train & Validation)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loaders = data_helper.get_target_dataloaders(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loaders}
        self.init_train_dataset_size = len(self.source_loader.dataset)
        print("Dataset size: train %d, val %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset)))

        # Logger
        self.log_frequency = (int)(len(self.source_loader) / 3)
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, train_all=True, nesterov=args.nesterov, adam = args.adam, decay_ratio = args.decay)
        self.n_classes = args.n_classes

        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_min_epoch(self):
        self.model.train() # 启用 BatchNormalization 和 Dropout

        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            class_loss_val, class_acc = self._do_min_iter(data, class_l, d_idx)
            self.log(
                {"Class Loss": class_loss_val},
                {"Class Acc": class_acc},
                data.shape[0] # for case where last batch size  <= self.batch_size
            )

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                if phase == 'test':
                    if self.args.source[0] in digits_datasets:
                        target_domains = ['svhn', 'mnist_m', 'syn', 'usps']
                    elif self.args.source[0] in pacs_datasets:
                        target_domains = [item for item in pacs_datasets if item != self.args.source[0]]

                    acc_sum = 0.0
                    for didx in range(len(loader)):
                        dkey = phase + '-' + target_domains[didx]

                        test_loader = loader[didx]
                        test_total = len(test_loader.dataset)
                        class_correct = self.do_test(test_loader)
                        class_acc = float(class_correct) / test_total

                        self.log_test(dkey, {"class": class_acc})
                        if dkey not in self.results.keys():
                            self.results[dkey] = torch.zeros(self.args.epochs)
                        self.results[dkey][self.current_epoch] = class_acc
                        acc_sum += class_acc
                    self.log_test(phase, {"class": acc_sum / len(loader)})
                    self.results[phase][self.current_epoch] = acc_sum / len(loader)
                else:
                    total = len(loader.dataset)
                    class_correct = self.do_test(loader)
                    class_acc = float(class_correct) / total

                    self.log_test(phase, {"class": class_acc})
                    self.results[phase][self.current_epoch] = class_acc

    def _do_min_iter(self, data, class_l, d_idx):
        criterion = nn.CrossEntropyLoss()
        data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)

        self.optimizer.zero_grad()

        last_features, class_logit = self.model(data)
        _, cls_pred = class_logit.max(dim=1)

        class_loss = criterion(class_logit, class_l)
        class_loss.backward()
        self.optimizer.step()

        class_loss_val = class_loss.item()
        class_acc = torch.sum(cls_pred == class_l.data).item()

        self.current_iter +=1
        return class_loss_val, class_acc

    def _do_max_phase(self, T_max, X_n, Y_n):
        # shape of X_n: B X H X W X C
        # from the Tensorflow implementation given by paper author, we can see N = BatchSize
        X_n, Y_n = X_n.to(self.device), Y_n.to(self.device)
        class_criterion = nn.CrossEntropyLoss()
        semantic_distance_criterion = nn.MSELoss()
        max_optimizer = optim.SGD([X_n.requires_grad_()], lr=self.args.adv_learning_rate)

        self.model.eval()

        init_feature = None
        for i in range(T_max):
            max_optimizer.zero_grad()

            last_features, class_logit = self.model(X_n)
            if i == 0:
                init_feature = last_features.clone().detach()
            _, cls_pred = class_logit.max(dim=1)

            class_loss = class_criterion(class_logit, Y_n)
            feature_loss = semantic_distance_criterion(last_features, init_feature)
            adv_loss = self.args.gamma * feature_loss - class_loss

            adv_loss.backward()
            max_optimizer.step()

            # 解除变量引用与实际值的指向关系
            del class_loss, feature_loss, class_logit, last_features
        return X_n.to('cpu').detach(), cls_pred.to('cpu').detach().flatten().tolist(), adv_loss

    def do_training(self):
        self.current_epoch = 0
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        self.total_iters_cur_epoch = len(self.source_loader)

        for k in range(self.args.K):
            self.current_iter = 0
            class_loss, class_acc = 0.0, 0.0
            for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
                if self.current_iter == self.args.T_min:
                    print("Min-phase ended, class loss: %g, class acc: %g. %d-th Max-phase started!"
                          % (class_loss, class_acc, k))
                    data_adv, labels_adv, loss_adv = self._do_max_phase(self.args.T_max, data, class_l)
                    self.source_loader = data_helper.append_adversarial_samples(
                        self.args, self.source_loader, data_adv, labels_adv)
                    print("%d-th Max-phase ended, Adv loss: %g" % (k, -1 * loss_adv))
                    break
                class_loss, class_acc = self._do_min_iter(data, class_l, d_idx)

        print("%d rounds of adeversarial procedure ended! Start classifiction training!" % self.args.K)
        
        self.current_iter = 0
        cur_train_dataset_size = len(self.source_loader.dataset)
        print("New Training Dataset Size %d after Adv Procedure! %d adversarial images added" 
              % (cur_train_dataset_size, cur_train_dataset_size - self.init_train_dataset_size))
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_min_epoch()
            self.current_epoch+=1
            self.current_iter=0

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx__val_best = val_res.argmax()
        idx_test_best = test_res.argmax()

        print("Best test acc: %g in epoch: %d" % (test_res.max(), idx_test_best+1))
        return self.model

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, class_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            _, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def log(self, losses_dic, correct_samples_dic, total_samples_cur_batch):
        losses_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses_dic.items()])
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples_cur_batch)) for k, v in correct_samples_dic.items()])
        if self.current_iter % self.log_frequency == 0:
            print("%d/%d of epoch %d/%d %s - %s [bs:%d]" % (self.current_iter, self.total_iters_cur_epoch,
                                                            self.current_epoch + 1,self.args.epochs, losses_string,
                                                            acc_string, total_samples_cur_batch
                                                           )
                 )

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))
    def test_func(self):
        fake_data = torch.ones(128,3,222,222)
        fake_labels = torch.ones(128,1)
        loader = data_helper.append_adversarial_samples(self.args, self.source_loader, fake_data, fake_labels)
        for it, ((data, class_l), _) in enumerate(loader):
            print(data.shape)


def main():
    torch.manual_seed(9963)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(9963)
    random.seed(9963)

    args = get_args()
    torch.cuda.set_device(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()
    #trainer.test_func()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # 设为 True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    main()
