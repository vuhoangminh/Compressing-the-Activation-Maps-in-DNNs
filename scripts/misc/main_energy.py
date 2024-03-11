import comet_ml as comet

import argparse
import os
import random
import warnings
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import engine.training as training
import engine.utils.train_utils as train_utils
import engine.utils.print_utils as print_utils
import engine.logger as logger

from pprint import pprint

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
# parser.add_argument('--path_opt', default='database/options/cifar10/mobilenet_compressed_two_layers_db3_0.0.yaml', type=str,
parser.add_argument(
    "--path_opt",
    # default="database/options/mnist/mnistnet_compressed_db5_0_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db5_0_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db5_0.999_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_one_layer_db3_0.99_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_one_layer_haar_0.99_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_one_layer_haar_0.99999_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_one_layer_db3_0.1_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db3_0.1_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_sym3_1_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db3_0.99_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db3_0.1_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_db3_0.9999_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_one_layer_db3_0_3.yaml",
    # default="database/options/mnist/mnistnet_compressed_sym5_0.1_3.yaml",
    default="database/options/mnist/mnistnet_compressed_rbio1.3_0.999999_3.yaml",
    # default="database/options/mnist/mnistnet.yaml",
    # default="database/options/mnist/mnistnet_freeze.yaml",
    type=str,
    # parser.add_argument('--path_opt', default='database/options/mnist/mnistnet.yaml', type=str,
    help="path to a yaml options file",
)
parser.add_argument("--dir_logs", type=str, help="dir logs")
parser.add_argument("-a", "--arch", metavar="ARCH")
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 0)",
)
parser.add_argument(
    "--epochs", type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="learning_rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--patience", type=int, help="patience in training")
parser.add_argument("--early_stopping", type=int, help="early stopping in training")
parser.add_argument(
    "--wd",
    "--weight_decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print_freq",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="ckpt",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--save_model",
    default=True,
    type=train_utils.str2bool,
    help="able or disable save model and optim state",
)
parser.add_argument(
    "--save_all_from",
    type=int,
    help="""delete the preceding checkpoint until an epoch,"""
    """ then keep all (useful to save disk space)')""",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--empty_memory_cache",
    default=1,
    type=int,
    help="empty memory cache after each step?",
)
################################################
parser.add_argument(
    "-ho",
    "--help_opt",
    dest="help_opt",
    action="store_true",
    help="show selected options before running",
)

parser.add_argument("--is_test", type=int, default=1)


best_acc1 = 0


def main():
    args = parser.parse_args()
    #########################################################################################
    # Set datasets options
    #########################################################################################
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    #########################################################################################
    # Create options
    #########################################################################################
    options = {
        "logs": {"dir_logs": args.dir_logs},
        "model": {"arch": args.arch},
        "optim": {
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "early_stopping": args.early_stopping,
        },
    }
    if args.path_opt is not None:
        with open(args.path_opt, "r") as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        options = train_utils.update_values(options, options_yaml)
    print_utils.print_separator()
    print("## args")
    pprint(vars(args))
    print_utils.print_separator()
    print("## options")
    pprint(options)
    if args.help_opt:
        return

    if args.gpu is not None:
        print_utils.print_separator()
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    if args.pretrained:
        print_utils.print_separator()
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print_utils.print_separator()
        print("=> creating model '{}'".format(options["model"]["arch"]))
        # model = models.__dict__[args.arch]()
        model = training.build_model(options)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), options['optim']['learning_rate'],
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=options["optim"]["learning_rate"], betas=(0.9, 0.999)
    )

    #########################################################################################
    # args.resume: resume from a checkpoint OR create logs directory
    #########################################################################################
    # options['logs']['old_dir_logs'] = options['logs']['dir_logs']
    if args.empty_memory_cache:
        options["logs"]["dir_logs"] = (
            options["logs"]["dir_logs"] + "_torch.cuda.empty_cache"
        )

    exp_logger = None
    is_loadable_resume = False
    is_training = True
    if args.resume:
        (
            args.start_epoch,
            best_acc1,
            exp_logger,
            is_loadable_resume,
        ) = training.load_checkpoint(
            model, optimizer, os.path.join(options["logs"]["dir_logs"], args.resume)
        )
    if not is_loadable_resume:
        # Or create logs directory
        os.system("mkdir -p " + options["logs"]["dir_logs"])
        path_new_opt = os.path.join(
            options["logs"]["dir_logs"], os.path.basename(args.path_opt)
        )
        path_args = os.path.join(options["logs"]["dir_logs"], "args.yaml")
        with open(path_new_opt, "w") as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if exp_logger is None:
        #  Set loggers
        exp_name = os.path.basename(options["logs"]["dir_logs"])  # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters("train", training.make_meters())
        exp_logger.add_meters("test", training.make_meters())
        exp_logger.info["model_params"] = train_utils.params_count(model)
        print("Model has {} parameters".format(exp_logger.info["model_params"]))

    if is_loadable_resume:
        if len(exp_logger.logged["train"]["loss"]) >= args.start_epoch + 1:
            is_training = False
            print("Model was fully trained. Will skip!")

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader, train_sampler = training.setup_dataloader_transform(
        args=args, options=options
    )

    if args.evaluate and is_training:
        path_logger_json = os.path.join(options["logs"]["dir_logs"], "logger.json")

        acc1 = training.validiate(
            val_loader,
            model,
            criterion,
            exp_logger,
            args,
            options,
            print_freq=args.print_freq,
        )
        exp_logger.to_json(path_logger_json)
        return

    #########################################################################################
    #  Begin training on train/val or trainval/test
    #########################################################################################
    is_test = args.is_test
    if is_test:
        experiment = None
    else:
        experiment = comet.Experiment(
            api_key="AgTGwIoRULRgnfVR5M8mZ5AfS",
            project_name="compress",
            workspace="vuhoangminh",
        )
        experiment.log_parameters(training.flatten(options))

    if is_training:
        print_utils.print_separator()
        print("=> Start Training")
        if experiment is not None:
            with experiment.train():
                E_epoch = None
                T_epoch = None
                for epoch in range(args.start_epoch, options["optim"]["epochs"]):
                    if args.distributed:
                        train_sampler.set_epoch(epoch)
                    training.adjust_learning_rate(
                        optimizer, epoch, options["optim"]["learning_rate"]
                    )

                    # train for one epoch
                    training.train(
                        train_loader,
                        model,
                        criterion,
                        optimizer,
                        exp_logger,
                        epoch,
                        experiment,
                        args,
                        options,
                        E_epoch,
                        T_epoch,
                        print_freq=args.print_freq,
                    )

                    # evaluate on validation set
                    with experiment.validate():
                        acc1 = training.validiate(
                            val_loader,
                            model,
                            criterion,
                            exp_logger,
                            args,
                            options,
                            epoch=epoch,
                            print_freq=args.print_freq,
                        )
                        experiment.log_metric("acc1", acc1)

                    # remember best acc@1 and save checkpoint
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)

                    if not args.multiprocessing_distributed or (
                        args.multiprocessing_distributed
                        and args.rank % ngpus_per_node == 0
                    ):
                        training.save_checkpoint(
                            {
                                "epoch": epoch,
                                "arch": options["model"]["arch"],
                                "best_acc1": best_acc1,
                                "exp_logger": exp_logger,
                            },
                            model.state_dict(),
                            optimizer.state_dict(),
                            options["logs"]["dir_logs"],
                            args.save_model,
                            args.save_all_from,
                            is_best,
                        )

        else:
            E_epoch = None
            T_epoch = None
            for epoch in range(args.start_epoch, options["optim"]["epochs"]):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                training.adjust_learning_rate(
                    optimizer, epoch, options["optim"]["learning_rate"]
                )

                # train for one epoch
                E_epoch, T_epoch = training.train(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    exp_logger,
                    epoch,
                    experiment,
                    args,
                    options,
                    E_epoch,
                    T_epoch,
                    print_freq=args.print_freq,
                )

                # evaluate on validation set
                acc1 = training.validiate(
                    val_loader,
                    model,
                    criterion,
                    exp_logger,
                    args,
                    options,
                    epoch=epoch,
                    print_freq=args.print_freq,
                )

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
                ):
                    training.save_checkpoint(
                        {
                            "epoch": epoch,
                            "arch": options["model"]["arch"],
                            "best_acc1": best_acc1,
                            "exp_logger": exp_logger,
                        },
                        model.state_dict(),
                        optimizer.state_dict(),
                        options["logs"]["dir_logs"],
                        args.save_model,
                        args.save_all_from,
                        is_best,
                    )


if __name__ == "__main__":
    main()

    """
    @TODO
    number of compressed layers: 1, 2
    compression ratio: 0.01, 0.03, 0.1, 0.3, 0.6, 0.99
    wave: 'bior1.1', 'bior2.2', 'coif3', 'db1', 'db3', 'db5', 'rbio1.3', 'sym3'
    number of levels is also a hyper-parameter: 2, 3, 4, 5, 6
    training time per epoch (Fig 7) + confidence intervals

    @BUG:
    [fixed] allocated memory keeps increasing
    """
