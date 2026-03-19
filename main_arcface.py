import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils.utils as utils
import utils.config as config
from train_arcface import train, evaluate
import modules.base_model_arcface as base_model
from utils.dataset import Dictionary, VQAFeatureDataset
from utils.losses import Plain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of running epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate for adamax')
    parser.add_argument('--loss-fn', type=str, default='Plain',
                        help='chosen loss function')
    parser.add_argument('--num-hid', type=int, default=1024,
                        help='number of dimension in last layer')
    parser.add_argument('--model', type=str, default='baseline_newatt',
                        help='model structure')
    parser.add_argument('--base-model', type=str, default='BAN',
                        help='backbone identifier used by dataset branches')
    parser.add_argument('--feat-dim', type=int, default=1024,
                        help='visual feature dimension for optional MEVF branch')
    parser.add_argument('--name', type=str, default='exp0.pth',
                        help='saved model name')
    parser.add_argument('--name-new', type=str, default=None,
                        help='combine with fine-tune')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size')
    parser.add_argument('--fine-tune', action='store_true',
                        help='fine tuning with our loss')
    parser.add_argument('--resume', action='store_true',
                        help='whether resume from checkpoint')
    parser.add_argument('--not-save', action='store_true',
                        help='do not overwrite the old model')
    parser.add_argument('--test', dest='test_only', action='store_true',
                        help='test one time')
    parser.add_argument('--eval-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument("--lambda1", type=float, default=1,
                        help='hyperparameter1 for the loss function')
    parser.add_argument("--lambda2", type=float, default=1,
                        help='hyperparameter2 for the loss function')
    parser.add_argument("--m", type=float, default=1.0,
                        help='hyperparameter3 for the loss function')
    parser.add_argument(
        "--dataset",
        default="slake",
        choices=["slake", "slake-cp", "vqa-rad", "vqa-rad-cp", "omni", "omni-cp", "vqacp-v2", "vqace", "pmca", "pmca-cp"],
        help="choose dataset",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--aux-warmup-epochs", type=int, default=5,
                        help="linearly warm up auxiliary losses (SupCon/DDC/ECC) over N epochs")
    parser.add_argument("--aux-loss-weight", type=float, default=1.0,
                        help="max weight multiplier for auxiliary losses after warmup")
    parser.add_argument("--max-aux-loss", type=float, default=5.0,
                        help="cap the scaled auxiliary-loss contribution per batch to improve stability")
    parser.add_argument('--no-ddc', action='store_true',
                        help='disable DDC auxiliary loss (KL-based term) for ablation')
    parser.add_argument('--no-ecc', action='store_true',
                        help='disable ECC auxiliary loss (energy constraint term) for ablation')
    args = parser.parse_args()
    return args


def resolve_run_paths(args):
    if 'log' not in args.name:
        args.name = 'logs/' + args.name

    base_dir = os.path.dirname(args.name)
    if not base_dir:
        base_dir = 'logs'
    base_name = os.path.basename(args.name)

    # New trainings create an isolated run directory; resume keeps original path.
    if args.resume or args.test_only or args.eval_only:
        run_dir = base_dir
        checkpoint_base = args.name
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = os.path.join(base_dir, '{}-{}'.format(base_name, run_id))
        checkpoint_base = os.path.join(run_dir, base_name)

    os.makedirs(run_dir, exist_ok=True)
    return checkpoint_base, run_dir


def checkpoint_meta(path):
    if not os.path.exists(path):
        return None

    stat = os.stat(path)
    return {
        'path': path,
        'size_mb': stat.st_size / (1024 * 1024),
        'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }


def write_run_report(report_path, args, device, run_status, best_epoch, best_val_score, latest_epoch, best_path, latest_path, interrupt_path=None):
    lines = [
        '# Run Report',
        '',
        '## Run Info',
        '- status: {}'.format(run_status),
        '- generated_at: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        '- dataset: {}'.format(args.dataset),
        '- model: {}'.format(args.model),
        '- device: {}'.format(device),
        '- epochs_target: {}'.format(args.epochs),
        '- latest_epoch: {}'.format(latest_epoch),
        '- best_epoch: {}'.format(best_epoch),
        '- best_val_score: {:.6f}'.format(best_val_score if best_val_score != float('-inf') else float('nan')),
        '',
        '## Checkpoints',
        '| type | exists | path | size_mb | modified |',
        '| --- | --- | --- | --- | --- |'
    ]

    for ckpt_type, ckpt_path in [('best', best_path), ('latest', latest_path), ('interrupt', interrupt_path)]:
        if ckpt_path is None:
            lines.append('| {} | no | - | - | - |'.format(ckpt_type))
            continue

        meta = checkpoint_meta(ckpt_path)
        if meta is None:
            lines.append('| {} | no | {} | - | - |'.format(ckpt_type, ckpt_path))
        else:
            lines.append('| {} | yes | {} | {:.2f} | {} |'.format(
                ckpt_type,
                meta['path'],
                meta['size_mb'],
                meta['mtime']
            ))

    lines.extend([
        '',
        '## Args',
        '```json',
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        '```',
        ''
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_history(history_json_path, history_csv_path, history_rows):
    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history_rows, f, ensure_ascii=False, indent=2)

    with open(history_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'eval_score'])
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    config.dataset = dataset
    config.update_paths(args.dataset)
    if dataset in ['vqa-rad', 'vqa-rad-cp', 'slake', 'slake-cp']:
        args.batch_size = 64
    print(args)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

    if args.test_only or args.fine_tune or args.eval_only:
        args.resume = True

    args.name, run_dir = resolve_run_paths(args)
    report_path = os.path.join(run_dir, 'run-report.md')

    if args.resume and not args.name:
        raise ValueError("Resuming requires folder name!")
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))

    # ------------------------DATASET CREATION--------------------
    dictionary = Dictionary.load_from_file(config.dict_path)
    if args.test_only:
        eval_dset = VQAFeatureDataset('test', dictionary, args)
    else:
        train_dset = VQAFeatureDataset('train', dictionary, args)
        eval_dset = VQAFeatureDataset('test', dictionary, args)
    # if config.train_set == 'train+val' and not args.test_only:
    #     train_dset = train_dset + eval_dset
    #     eval_dset = VQAFeatureDataset('test', dictionary)
    # if args.eval_only:
    #     eval_dset = VQAFeatureDataset('val', dictionary)

    tb_count = 0
    writer = SummaryWriter() # for visualization

    if not config.train_set == 'train+val' and 'LM' in args.loss_fn:
        utils.append_bias(train_dset, eval_dset, len(eval_dset.label2ans))

    # ------------------------MODEL CREATION------------------------
    constructor = 'build_{}'.format(args.model)
    model, metric_fc = getattr(base_model, constructor)(eval_dset, args.num_hid)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('using device:', device)
    model = model.to(device)
    metric_fc = metric_fc.to(device)
    model.w_emb.init_embedding(config.glove_embed_path)

    # model = nn.DataParallel(model).cuda()
    optim = torch.optim.Adamax([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr)

    if args.loss_fn == 'Plain':
        loss_fn = Plain()
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))

    # ------------------------STATE CREATION------------------------
    eval_score, best_val_score, start_epoch, best_epoch = 0.0, float('-inf'), 0, 0
    tracker = utils.Tracker()
    if args.resume:
        model.load_state_dict(logs['model_state'])
        metric_fc.load_state_dict(logs['margin_model_state'])
        optim.load_state_dict(logs['optim_state'])
        if 'loss_state' in logs:
            loss_fn.load_state_dict(logs['loss_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_val_score = logs['best_val_score']
        if args.fine_tune:
            print('best accuracy is {:.2f} in baseline'.format(100 * best_val_score))
            args.epochs = start_epoch + 10 # 10 more epochs
            for params in optim.param_groups:
                params['lr'] = config.ft_lr

            # if you want save your model with a new name
            if args.name_new:
                if 'log' not in args.name_new:
                    args.name = 'logs/' + args.name_new
                else:
                    args.name = args.name_new

    eval_loader = DataLoader(eval_dset,
                    args.batch_size, shuffle=False, num_workers=4)

    if args.test_only or args.eval_only:
        model.eval()
        metric_fc.eval()
        evaluate(model, metric_fc, eval_loader, write=True)
    else:
        train_loader = DataLoader(
            train_dset, args.batch_size, shuffle=True, num_workers=4)

        interrupted = False
        current_epoch = start_epoch
        latest_epoch = start_epoch
        run_status = 'running'
        history_rows = []
        history_json_path = os.path.join(run_dir, 'history.json')
        history_csv_path = os.path.join(run_dir, 'history.csv')

        latest_path = args.name + '.latest'
        best_path = args.name
        interrupt_path = args.name + '.interrupt'

        write_run_report(
            report_path,
            args,
            device,
            run_status,
            best_epoch,
            best_val_score,
            latest_epoch,
            best_path,
            latest_path,
            None
        )

        try:
            for epoch in range(start_epoch, args.epochs):
                current_epoch = epoch
                print("training epoch {:03d}".format(epoch))
                tb_count, train_loss, train_acc = train(model, metric_fc, optim, train_loader, loss_fn, tracker, writer, tb_count, epoch, args)

                if not (config.train_set == 'train+val' and epoch in range(args.epochs - 3)):
                    # save for the last three epochs
                    write = True if config.train_set == 'train+val' else False
                    print("validating after epoch {:03d}".format(epoch))
                    model.train(False)
                    metric_fc.train(False)
                    eval_score = evaluate(model, metric_fc, eval_loader, epoch, write=write)
                    model.train(True)
                    metric_fc.train(True)
                    print("eval score: {:.2f} \n".format(100 * eval_score))

                results = {
                    'epoch': epoch + 1,
                    'best_val_score': best_val_score,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                    'margin_model_state': metric_fc.state_dict()
                }

                if not args.not_save:
                    torch.save(results, latest_path)

                if eval_score >= best_val_score:
                    best_val_score = eval_score
                    best_epoch = epoch
                    results['best_val_score'] = best_val_score
                    if not args.not_save:
                        torch.save(results, best_path)

                latest_epoch = epoch + 1

                history_rows.append({
                    'epoch': latest_epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'eval_score': float(eval_score)
                })
                write_history(history_json_path, history_csv_path, history_rows)

                write_run_report(
                    report_path,
                    args,
                    device,
                    run_status,
                    best_epoch,
                    best_val_score,
                    latest_epoch,
                    best_path,
                    latest_path,
                    None
                )

        except KeyboardInterrupt:
            interrupted = True
            run_status = 'interrupted'
            print("training interrupted by user, preparing interrupt checkpoint...")
            if not args.not_save:
                results = {
                    'epoch': current_epoch + 1,
                    'best_val_score': best_val_score,
                    'model_state': model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'loss_state': loss_fn.state_dict(),
                    'margin_model_state': metric_fc.state_dict()
                }
                torch.save(results, interrupt_path)
                print("interrupt checkpoint saved to {}".format(interrupt_path))

            write_run_report(
                report_path,
                args,
                device,
                run_status,
                best_epoch,
                best_val_score,
                current_epoch + 1,
                best_path,
                latest_path,
                interrupt_path if not args.not_save else None
            )

        if best_val_score == float('-inf'):
            print("no evaluation result produced before stop")
        else:
            print("best accuracy {:.2f} on epoch {:03d}".format(
                100 * best_val_score, best_epoch))

        if interrupted:
            print("training exited early after user interruption")
        else:
            run_status = 'finished'
            write_run_report(
                report_path,
                args,
                device,
                run_status,
                best_epoch,
                best_val_score,
                latest_epoch,
                best_path,
                latest_path,
                None
            )
