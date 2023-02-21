# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from util.dictionary import build_dictionary
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, SubsetRandomSampler
from engine import evaluate, train_one_epoch
# from models import build_model
from playground import build_all_model
from timm.utils import NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--filter_weight_decay', action='store_true')# zychen
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--amp_train', action='store_true', help='amp fp16 training or not')
    parser.add_argument('--eval_epoch', default=5, type=int)

    # Pix2Seq
    parser.add_argument('--model', type=str, default="pix2seq",
                        help="specify the model from playground")
    parser.add_argument('--pix2seq_lr', action='store_true', help='use warmup linear drop lr')
    parser.add_argument('--large_scale_jitter', action='store_true', help='large scale jitter')
    parser.add_argument('--rand_target', action='store_true',
                        help="randomly permute the sequence of input targets")
    parser.add_argument('--pred_eos', action='store_true', help='use eos token instead of predicting 100 objects')
    # zychen for augmentation
    parser.add_argument('--aug_scale_min', default=0.3, type=float)
    parser.add_argument('--aug_scale_max', default=2.0, type=float)
    parser.add_argument('--color_distortion', action='store_true', dest='color_distortion')
    parser.add_argument('--no_color_distortion', action='store_false', dest='color_distortion')
    parser.set_defaults(color_distortion=True)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--pretrained', action='store_true',
                        help="whether to use pretrain model or not")
    parser.add_argument('--position_embedding', default='mine', type=str, choices=('sine', 'learned', 'mine'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help="DropPath applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--classifier_norm', action='store_true')

    # * Loss coefficients
    parser.add_argument('--eos_coef', default=1.0, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--loss_type', default='ce', type=str, choices=('ce', 'ce_specific', 'focal'))
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--remove_empty_annotations', action='store_true') # zychen

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    # zychen for sampling
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.4, type=float)
    parser.add_argument('--temperature', default=1., type=float)
    # zychen for evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eos_bias', default=0., type=float)
    parser.add_argument('--eval_p', default=0., type=float)

    # zychen added
    parser.add_argument('--max_input_size', default=1333, type=int)
    parser.add_argument('--max_objects', default=100, type=int)
    parser.add_argument('--num_bins', default=2000, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # zychen added
    parser.add_argument('--return_intermediate_dec', action='store_true')
    parser.add_argument('--query_pos', action='store_true')
    parser.add_argument('--drop_cls', default=0., type=float)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.dictionary = build_dictionary(args)
    model, criterion, postprocessors = build_all_model[args.model](args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build param_dicts
    skip_list = ["transformer.vocal_embed.weight", "transformer.det_embed.weight", "transformer.query_pos.weight", "backbone.1.pos_embed"]
    weight_backbone, bias_backbone, weight_head, bias_head = [], [], [], []
    for n, p in model_without_ddp.named_parameters():
        if args.filter_weight_decay:
            if "backbone.0" in n:
                if len(p.shape) == 1 or n.endswith(".bias"):
                    bias_backbone.append(p)
                else:
                    weight_backbone.append(p)
            else:
                if len(p.shape) == 1 or n.endswith(".bias") or n in skip_list:
                    bias_head.append(p)
                else:
                    weight_head.append(p)
        else:
            if "backbone" in n:
                weight_backbone.append(p)
            else:
                weight_head.append(p)
    param_dicts = [
        {"params": weight_backbone, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": bias_backbone, "lr": args.lr_backbone, "weight_decay": 0.},
        {"params": weight_head, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": bias_head, "lr": args.lr, "weight_decay": 0.},
    ]
    for item in param_dicts:
        print(f"{len(item['params'])} items with lr {item['lr']}, weight decay {item['weight_decay']}")
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.pix2seq_lr:
        lr_scheduler = utils.WarmupLinearDecayLR(
            optimizer,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            end_epoch=args.epochs,
            final_lr_factor=0.01)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    loss_scaler = NativeScaler() if args.amp_train else utils.NoScaler()

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=partial(utils.collate_fn, fix_input=args.max_input_size), num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=partial(utils.collate_fn, fix_input=args.max_input_size), num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    cur_ap = max_ap = 0.0
    if (not args.resume) and os.path.exists(str(output_dir/"checkpoint.pth")):
        args.resume = str(output_dir/"checkpoint.pth")
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'epoch' in checkpoint:
            print(f"Resuming from Epoch {checkpoint['epoch']} ...")
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'ap' in checkpoint:
            cur_ap = checkpoint['ap']
            max_ap = checkpoint['max_ap']

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        print("Evaluation Results")
        print({f'test_{k}': v for k, v in test_stats.items()})
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            loss_scaler, args.clip_max_norm, amp_train=args.amp_train)
        lr_scheduler.step()

        if epoch % args.eval_epoch == args.eval_epoch -1 or epoch == (args.lr_drop - 1) or epoch == (args.epochs - 1):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            cur_ap = test_stats['coco_eval_bbox'][0]
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            test_stats = coco_evaluator = None

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if cur_ap > max_ap:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
                max_ap = cur_ap
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'ap': cur_ap,
                    'max_ap': max_ap,
                }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Seq training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
