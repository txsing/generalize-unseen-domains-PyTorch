import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES']='2, 3'

import torch
from torch import optim
#from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch Unseen training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
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
    parser.add_argument("--gamma", type=float, default=1.0, help="Higher val means stricter distance constraint")
    parser.add_argument("--adv_learning_rate", type=float, default=1.0, help="Learning rate for adversarial training")
    # nesterov 是一种梯度下降的方法
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")
    
    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        # Logger
        self.log_frequency = 10

        model = model_factory.get_network(args.network)(classes=args.n_classes)
        self.model = model.to(device)

        # The training dataset get divided into two parts (Train & Validation)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_target_dataloader(args)
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, train_all=True, nesterov=args.nesterov)
        self.n_classes = args.n_classes

        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_min_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train() # 启用 BatchNormalization 和 Dropout

        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l, d_idx = data.to(self.device), class_l.to(self.device), d_idx.to(self.device)

            self.optimizer.zero_grad()

            last_features, class_logit = self.model(data)
            if self.target_id is not None: # target domain 的图片用于 predict，因此不贡献 loss
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            else: # 对所有（包括打乱的）图片进行物种分类，target domain 只用于 predict
                class_loss = criterion(class_logit, class_l)

            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss
            loss.backward()
            self.optimizer.step()

            self.current_iter +=1
            self.log({"class": class_loss.item()},
                            {"class": torch.sum(cls_pred == class_l.data).item()},
                            data.shape[0] # Last batch size  <= self.batch_size
                            )

            # 解除变量引用与实际值的指向关系
            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total

                self.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def _do_max_epoch(self, T_max, X_n, Y_n):
        # shape of X_n: B X H X W X C
        # from the Tensorflow implementation given by paper author, we can see N = BatchSize
        X_n, Y_n = X_n.to(self.device), Y_n.to(self.device)
        class_criterion = nn.CrossEntropyLoss()
        semantic_distance_criterion = nn.MSELoss()
        max_optimizer = optim.SGD([X_n.requires_grad_()], lr=self.args.adv_learning_rate)

        self.model.eval()

        init_feature = None
        for i in range(T_max):
            X_n.data.clamp_(0,1)
            max_optimizer.zero_grad()

            last_features, class_logit = self.model(X_n)
            if i == 0:
                init_feature = last_features.clone().detach()
                print(init_feature.shape)

            class_loss = class_criterion(class_logit, Y_n)
            feature_loss = semantic_distance_criterion(last_features, init_feature)
            adv_loss = self.args.gamma * feature_loss - class_loss

            print(adv_loss)
            adv_loss.backward()
            max_optimizer.step()

            # 解除变量引用与实际值的指向关系
            del adv_loss, class_loss, feature_loss, class_logit, last_features
        X_n.data.clamp_(0,1)
        return X_n

    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        self.total_iters_cur_epoch = len(self.source_loader)
        self.current_epoch = 0
        self.current_iter = 0

        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_min_epoch()

            self.current_epoch+=1
            self.current_iter=0

        for it, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data_adv = self._do_max_epoch(6, data, class_l)
            print(torch.equal(data, data_adv.to('cpu')))
            break

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
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
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (self.current_iter, self.total_iters_cur_epoch, self.current_epoch, self.args.epochs, losses_string,
                                                                acc_string, total_samples_cur_batch))

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # 设为 True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    main()
