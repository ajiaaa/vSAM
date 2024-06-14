import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
import sys; sys.path.append("")
from utility.sharpness import SHARPNESS_NOISE
from model.resnet import ResNet18 as resnet18
from model.PyramidNet import PyramidNet as PYRM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--alpha", default=0.4, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")

    parser.add_argument("--solver_num", default=1, type=int)
    parser.add_argument("--rho_test_list", default=[0.001, 0.005, 0.01, 0.05, 0.1])

    args = parser.parse_args()

    #dataset = Cifar(args.batch_size, args.threads)
    #log = Log(log_each=10)

    # test_model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10)# .cuda()
    test_model = resnet18(num_classes=10).cuda()
    model_i_path = r'C:\Users\Jarvis\Desktop\save_model\cifar100\resnet\sgd\sam_ResNet_cifar100_epoch197'
    test_model = torch.load(model_i_path + '.pth', map_location='cuda:0')


    state = {
        'state_dict': test_model.state_dict()
    }
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    torch.save(state, model_i_path + '.t7')
