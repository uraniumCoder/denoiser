# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys

from pesq import pesq
from pystoi import stoi
import torch

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

from prettytable import PrettyTable


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    'denoiser.evaluate',
    description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")
parser.add_argument('--convert', action='store_true', help='Convert to 16kHz before evaluation')
parser.add_argument('--updates', type=int, default=100, help='Number of times to log progress')
parser.add_argument('--bit_width', default="32", help='quantize, 8, 4, 2')


def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = args.updates

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate, convert=args.convert)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, updates=updates, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    # print('cpu')
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    # print('cuda')
                    estimate = get_estimate(model, noisy, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates=updates, name="Eval metrics"):
            pesq_i, stoi_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i

    metrics = [total_pesq, total_stoi]
    pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}.'))
    return pesq, stoi

def get_size(args):
    nparams = _count_parameters(model)
    model_size = get_model_size(model, int(args.bit_width))
    return nparams, model_size

def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=model.sample_rate)


def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=sr)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=sr)
    return pesq_i, stoi_i


def _count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def get_model_size(model, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    logger.info(f"Total size in MiB: {num_elements * data_width/MiB:.2f} MiB")
    return num_elements * data_width


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    nparams, model_size = get_size(args)
    json.dump({'pesq': pesq, 'stoi': stoi, 'nparams': nparams, 'model size': model_size}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
