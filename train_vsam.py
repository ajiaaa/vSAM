import argparse
import torch
import sys
sys.path.append("..")
from model.wideresnet import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar, Cifar100
from utility.log import Log
from utility.initialize import initialize
import time
from tqdm import tqdm
from vsam import vSAM, CosineScheduler
from model.mobilenetv1 import MobileNetV1
from model.resnet import ResNet18 as resnet18
from model.resnet import ResNet50 as resnet50
from model.PyramidNet import PyramidNet as PYRM
import logging
import os
import random
from data.tiny_imagenet import TinyImageNet
import torchvision
def write_to_file(filename, data):
    f = open(filename, 'a')
    f.write(data)
    f.write('\n')
    f.close()


def loss_fn(predictions, targets):
    return smooth_crossentropy(predictions, targets, smoothing=0.1).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default='', type=str)
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--win_spl", default=[50, 5], help="[Window, number of blocks]")
    parser.add_argument("--max_p", default=0.8, help="[max of p]")
    parser.add_argument("--alpha", default=[0.13, 0.13], help="[]")
    parser.add_argument("--beta", default=0.7, type=float, help="beta")
    parser.add_argument("--dataset", default='cifar100', type=str, help="cifar10, cifar100")
    parser.add_argument("--model", default='resnet18', type=str, help="resnet18, wideresnet, pyramidnet")
    parser.add_argument("--myoptimizer", default='vsam', type=str, help="vsam")
    parser.add_argument("--last_layer", default=10, type=int, help="")

    args = parser.parse_args()

    seed = random.randint(1, 2000)
    print('seed:', seed)
    initialize(args, seed=seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))


    # 创建当前时间文件夹
    os.mkdir('save/' + stime)
    os.mkdir('save/' + stime + '/save_model')

    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('save/' + stime + '/log_' + stime + ".txt")
    handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logging.info('*' * 50)
    for arg, value in sorted(vars(args).items()):
        logging.info("%s: %r", arg, value)
    logging.info('*' * 50)


    if args.dataset == 'cifar10':
        dataset = Cifar(args.batch_size, args.threads)
        class_num = 10
    elif args.dataset == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads)
        class_num = 100
    elif args.dataset == 'tinyimagenet':
        class_num = 200
        dataset = TinyImageNet(args.batch_size, args.threads)

    log = Log(log_each=10)

    model = {
        'resnet18': resnet18(num_classes=class_num).to(device),
        'wideresnet': WideResNet(28, 10, args.dropout, in_channels=3, labels=class_num).to(device),
        'pyramidnet': PYRM('cifar' + str(class_num), 110, 270, class_num, False).to(device)
    }[args.model]

    for name, p in model.named_parameters():
        print(name)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    scheduler = CosineScheduler(T_max=args.epochs * len(dataset.train), max_value=args.learning_rate, min_value=0.0, optimizer=base_optimizer)

    optimizer = vSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                     rho=args.rho, adaptive=args.adaptive, hp_win=args.win_spl,
                    max_p=args.max_p, hp1=args.alpha, hp2=args.beta, last_layer = args.last_layer)

    whole_train_time = 0
    sampling_times = 0
    best_acc = 0

    for epoch in range(args.epochs):
        logger.info('epoch:' + str(epoch))
        model.train()
        optimizer.epoch = epoch
        a_cosine_sg_og, a_cosine_sg_ng, a_cosine_og_ng, a_cosine_og_sng = 0, 0, 0, 0
        time_start = time.time()
        for batch in tqdm(dataset.train):
            inputs, targets = batch[0].to(device), batch[1].to(device)  # (b.cuda() for b in batch)
            optimizer.set_closure(loss_fn, inputs, targets)
            predictions, sampling_times = optimizer.step()
            with torch.no_grad():
                scheduler.step()
        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        whole_train_time += time_sum
        logger.info('time:'+ str(whole_train_time))
        if epoch % 1 == 0 or args.epochs - epoch <= 20:
            model.eval()
            log.eval(len_dataset=len(dataset.test))
            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = batch[0].to(device), batch[1].to(device)  # (b.cuda() for b in batch)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
                loss_, acc_ = log.flush()
                logger.info('loss:' + str(loss_) + '   ' + 'acc:' + str(acc_))

        # if args.epochs - epoch <= 10 or epoch % 50 == 0:
        #     torch.save(model, 'save/' + stime + '/save_model' +'/epoch' + str(epoch) + '.pth')


    subject = '[' + args.dataset + '|' + args.model + '|' + args.myoptimizer + ']' + '_Result'
    content = 'acc:' + str(log.best_accuracy) + '   ' + 'train_time:' + str(
        whole_train_time) + '   ' + 'sampling_times:' + str(sampling_times)

    logger.info(content)

