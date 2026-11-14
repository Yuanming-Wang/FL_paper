# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# !/usr/bin/env python
import os
if os.path.isdir('/opt/maca'):
    os.environ.setdefault('MACA_PATH','/opt/maca')
import copy
import torch
import argparse
import hashlib
import time
import warnings
import numpy as np
import torchvision
import logging
import random
from collections.abc import Mapping
from pathlib import Path

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverS_scaffold import S_SCAFFOLD
# from flcore.servers.server_ours_huge import S_SCAFFOLD_huge
from flcore.servers.serverdpavg import FedDPAvg
from flcore.servers.serverFedMut import FedMut
from flcore.servers.serverheavg import HEFedAvg
from flcore.servers.server_privacy import AdapLDPFL, FedCEO, LDPFL, add_privacy_baseline_args
from flcore.servers.saveserver import exp

from flcore.trainmodel.models import *
from flcore.trainmodel.vgg16 import *
from flcore.trainmodel.VGG11 import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)
# hyper-params for Text tasks
vocab_size = 98635  # AG_News compatibility default
shakespeare_vocab_size = 80  # character vocabulary from language_utils.ALL_LETTERS
max_len = 200
emb_dim = 32


def run(args):
    source_root = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        source_hash.update(path.relative_to(source_root).as_posix().encode()+b'\0'+path.read_bytes())
    from utils.data_utils import data_root
    distribution = 'IID' if args.NIID>1 else f'Non-IID-{args.NIID}'
    partition = Path(data_root)/args.dataset/distribution
    data_hash = hashlib.sha256()
    for split in ('train','test'):
        for index in range(args.num_clients):
            relative = f'{split}/{index}.npz'
            data_hash.update(relative.encode()+b'\0'+(partition/relative).read_bytes())
    os.environ.update(FL_SOURCE_SHA256=source_hash.hexdigest(), FL_DATA_SHA256=data_hash.hexdigest())
    print(f'RUNTIME_PROVENANCE source_sha256={source_hash.hexdigest()} data_sha256={data_hash.hexdigest()}',flush=True)
    time_list = []
    model_str = args.model
    args.model_name = model_str
    task_vocab_size = shakespeare_vocab_size if "Shakespeare" in args.dataset else vocab_size
    print(args.model_name)


    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")

        start = time.time()
        args.device = "cuda:0" if args.device.startswith("cuda") else "cpu"
        # Generate args.model
        if model_str == "mlr":  # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":  # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "VGG11":
            args.model = VGG11(num_classes=args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=task_vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=task_vocab_size, hidden_size=emb_dim,
                                                   output_size=args.num_classes, num_layers=1,
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                   embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=task_vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=task_vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=task_vocab_size, d_model=64, nhead=4, nlayers=2,
                                          num_classes=args.num_classes, max_len=max_len,
                                          d_hid=128, pooling='last' if 'Shakespeare' in args.dataset else 'mean').to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        count = 0
        total_param = 0

        for param in args.model.parameters():
            count += 1
        total_layer = count
        # Decision-layer boundaries must follow the instantiated backbone.
        # The command-line default is kept for compatibility, but it cannot
        # describe every model (for example VGG11 and Transformer).
        args.total_layer = total_layer
        count = 0
        p_param = 0
        for param in args.model.parameters():
            if count >= total_layer - 6:
                p_param += param.numel()
            total_param += param.numel()
            count += 1
        print((p_param + total_param) * 4 / 1024 / 1024, "MB")

        # select algorithm
        if args.algorithm == "FedAvg":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "Adap-CTA":
            server = exp(args, i)

        elif args.algorithm == "DPFedAvg":
            server = FedDPAvg(args, i)

        elif args.algorithm == "FedMut":
            server = FedMut(args, i)

        elif args.algorithm == "FedPVR":
            server = S_SCAFFOLD(args, i)

        elif args.algorithm == "HEFedAvg":
            server = HEFedAvg(args, i)

        elif args.algorithm == "FedCEO":
            server = FedCEO(args, i)

        elif args.algorithm == "AdapLDP-FL":
            server = AdapLDPFL(args, i)

        elif args.algorithm == "LDP-FL":
            server = LDPFL(args, i)

        elif args.algorithm == "exp":
            server = exp(args, i)


        # elif args.algorithm == "ours":
        #     server = S_SCAFFOLD_huge(args, i)
        else:
            raise NotImplementedError

        if getattr(args, "initial_checkpoint", None):
            checkpoint = torch.load(args.initial_checkpoint, map_location=args.device)
            if isinstance(checkpoint, torch.nn.Module):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, Mapping):
                state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
            else:
                raise TypeError("initial checkpoint must be a module or state-dict mapping")
            server.global_model.load_state_dict(state_dict, strict=True)
            if hasattr(server, "initial_model_state"):
                server.initial_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in server.global_model.state_dict().items()
                }

        if getattr(args, 'resume_state', None):
            if args.algorithm not in ('LDP-FL','AdapLDP-FL','FedCEO','Adap-CTA','FedAvg','DPFedAvg','HEFedAvg','SCAFFOLD','FedMut','FedPVR') or not args.resume_curves:
                raise ValueError('state resume requires a supported main-table method and --resume-curves')
            server.restore_training_state(args.resume_state, args.resume_curves)
        server.train()

        # if hasattr(server, 'model'):
        #     try:
        #         model_params = sum(p.numel() for p in server.model.parameters())
        #         wandb.log({"model_parameters": model_params})  # 记录参数量
        #         print(f"Logged model parameters: {model_params}")
        #     except Exception as e:
        #         print(f"Could not log model size: {e}")

        time_list.append(time.time() - start)

        total_runtime = time.time() - start
        # wandb.log({"total_runtime_seconds": total_runtime})  # 记录总时间
        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    # reporter.report()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    total_start = time.time()
    parser = argparse.ArgumentParser()
    add_privacy_baseline_args(parser)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible client selection and initialization')
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="resnet")
    parser.add_argument('-mn', "--model_name", type=str, default="resnet")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=800)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="Adap-CTA")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1, help="Ratio of clients per round")
    parser.add_argument('-nv', "--noise_multiplier", type=float, default=0.00, help="noise variance")
    parser.add_argument('-NIID', "--NIID", type=float, default=0.2, help="d")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-ta', "--target_accuracy", type=float, default=0.6,
                        help="Target Global Accuracy ")
    parser.add_argument('-eg', "--eval_gap", type=int, default=5,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default=str(Path(__file__).resolve().parents[2] / "results/training/items"))
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-tl', "--total_layer", type=int, default=62)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedMut
    parser.add_argument('--mut_acc_rate', type=float, default=0.3,
                        help='FedMut control-rate coefficient')
    parser.add_argument('--fixed-decision-l', type=int, default=None,
                        help='Keep Adap-CTA correction depth fixed for sweeps')
    parser.add_argument('--allow-short-decision', action='store_true',
                        help='Diagnostic only: permit a correction depth below the formal minimum')
    parser.add_argument('--adapcta-control-warmup', type=int, default=0,
                        help='Adap-CTA only: initial rounds without control correction')
    parser.add_argument('--adapcta-balance-control', action='store_true',
                        help='Adap-CTA only: bound each control correction by its raw gradient norm')
    parser.add_argument("--mut_bound", type=int, default=20)
    parser.add_argument('--MutAlpha', type=float, default=2.0)


    parser.add_argument('-sf', "--safe_flag", type=bool, default=True,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument("--adaptive-capture-dir", default=None,
                        help="Optional directory for Adap-CTA observation bundles")
    parser.add_argument("--adaptive-capture-client", type=int, default=None,
                        help="Capture only this client id when adaptive capture is enabled")
    parser.add_argument("--adaptive-capture-round", type=int, default=None,
                        help="Capture only this zero-based training round")
    parser.add_argument("--initial-checkpoint", default=None,
                        help="Optional model checkpoint used as the initial global model")
    parser.add_argument('--resume-state', default=None, help='AdapLDP-FL training-state checkpoint')
    parser.add_argument('--resume-curves', default=None, help='Matching H5 curves for state resume')
    parser.add_argument("--reconstruction-eval", action="store_true",
                        help="Run the expensive DPFedAvg reconstruction evaluator")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    if os.name == 'posix' and args.device.startswith('cuda') and args.dataset != 'Shakespeare':
        import fcntl
        lock_dir = Path(__file__).resolve().parents[1] / 'state'
        lock_dir.mkdir(exist_ok=True)
        # Keep the handle alive through run(): independent retry queues share one image slot.
        image_slot = (lock_dir / f'image_cuda_{args.device_id}.lock').open('a')
        fcntl.flock(image_slot, fcntl.LOCK_EX)
    run(args)
