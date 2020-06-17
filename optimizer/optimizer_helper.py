from torch import optim
import math

def get_optim_and_scheduler(network, epochs, lr, train_all=True, nesterov=False, adam=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
 
    if adam: # refer to https://github.com/ricvolpi/generalize-unseen-domains
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay = 0.0005)
    else:
        optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = math.ceil(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler
