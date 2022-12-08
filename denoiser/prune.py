import copy
import torch
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from .pretrained import get_model, add_model_flags
from .utils import serialize_model
from .evaluate import evaluate
from .evaluate import parser as eval_parser

parser = argparse.ArgumentParser(
    'denoiser.prune',
    description='Prune model'
)
add_model_flags(parser)

# uniform pruning
parser.add_argument('--uniform', action='store_true', help='uniform pruning')
parser.add_argument('--p', type=float, help='pruning factor')
parser.add_argument('--output', type=str, help='output path')
parser.add_argument('--normalize_importance', action='store_true', help='normalize importance scores before summing')

# manual select pruning ratios
parser.add_argument('--target', type=str, help='json object of target number of channels')

# sensitivity scan
parser.add_argument('--sensitivity', action='store_true', help='sensitivity scan')
parser.add_argument('--scan_skips', action='store_true', help='scan the skip connections')
parser.add_argument('--scan_encoder', action='store_true', help='scan the encoder inner layers')
parser.add_argument('--scan_decoder', action='store_true', help='scan the decoder inner layers')

# plot sensitivity scan results
parser.add_argument('--plot_sensitivity', action='store_true', help='plot the sensitivity scan results')
parser.add_argument('--sensitivity_path', type=str, help='path to the sensitivity scan results')

# plot tradeoff results
parser.add_argument('--plot_tradeoff', action='store_true', help='plot the tradeoff curves')
parser.add_argument('--tradeoff_path', type=str, default='results/agg_pruning.txt', help='tradeoff data')
parser.add_argument('--include_designed', action='store_true', help='plot designed models')

def get_nchannels(model):
    """
    Get the number of channels of each layer in the model.
    """
    nchannels = {'encoder': [], 'decoder': [], 'lstm': []}
    nchannels['encoder'] = [[layer[0].in_channels, layer[2].in_channels] for layer in model.encoder]
    nchannels['decoder'] = [layer[2].in_channels for layer in model.decoder]
    nchannels['lstm'] = [model.lstm.lstm.input_size, model.lstm.lstm.hidden_size]

    return nchannels

def scale_down(n_channels, p):
    """
    Scale down the number of channels in the model by a factor of p.
    """
    tgt_nchannels = copy.deepcopy(n_channels)
    for i in range(len(tgt_nchannels['encoder'])):
        for k in range(2):
            if i==0 and k==0:
                continue # can't prune the input layer
            tgt_nchannels['encoder'][i][k] = int(tgt_nchannels['encoder'][i][k] * p)

    for i in range(len(tgt_nchannels['decoder'])):
        tgt_nchannels['decoder'][i] = int(tgt_nchannels['decoder'][i] * p)

    for i in range(2):
        tgt_nchannels['lstm'][i] = int(tgt_nchannels['lstm'][i] * p)
    
    return tgt_nchannels

def importance_selection(prev_convs, next_convs, n_channel, args):
    """
    Select the channels to keep based on importance scores.
    """
    importance = torch.zeros(next_convs[0].in_channels, device=next_convs[0].weight.device)
    for conv in next_convs:
        weight = conv.weight.detach()
        if isinstance(conv, torch.nn.Conv1d):
            layer_importance = torch.sqrt((weight**2).sum(dim=(0,2)))
        elif isinstance(conv, torch.nn.ConvTranspose1d):
            layer_importance = torch.sqrt((weight**2).sum(dim=(1,2)))
        if args is None or not args.normalize_importance:
            importance += layer_importance
        else:
            importance += layer_importance / torch.sqrt((layer_importance**2).sum())
    keep_channels = torch.argsort(importance, descending=True)[:n_channel].detach()
    return keep_channels

def prune(model, tgt_nchannels, args=None):
    """
    Prune the model using importance scores. (L2 on input channels)
    Currently doesn't support pruning the LSTM layer.
    """
    with torch.no_grad():
        cur_nchannels = get_nchannels(model)
        for i in range(len(model.encoder)):
            # prune first conv inputs layer (the skip channels)
            if i != 0:  # cannot prune input layer nor the lstm skip connection
                cur, tgt = cur_nchannels['encoder'][i][0], tgt_nchannels['encoder'][i][0]       
                assert cur >= tgt, "Cannot prune to more channels than exist."

                next_convs = model.encoder[i][0], model.decoder[-i][0]
                prev_convs = model.encoder[i-1][2], model.decoder[-i-1][2]
                keep_channels = importance_selection(prev_convs, next_convs, tgt, args)

                # last conv in encoder is glu layer, so we need to keep the channels in pairs
                duplicated_channels = torch.cat([keep_channels, keep_channels+cur])
                prev_convs[0].weight.set_(prev_convs[0].weight.detach()[duplicated_channels])
                prev_convs[0].bias.set_(prev_convs[0].bias.detach()[duplicated_channels])
                prev_convs[0].out_channels = tgt*2
                # last conv in decoder is transpose conv, so remove from 2nd dim
                prev_convs[1].weight.set_(prev_convs[1].weight.detach()[:, keep_channels])
                prev_convs[1].bias.set_(prev_convs[1].bias.detach()[keep_channels])
                prev_convs[1].out_channels = tgt
                for conv in next_convs:
                    conv.weight.set_(conv.weight.detach()[:, keep_channels])
                    conv.in_channels = tgt

            # prune middle connection in encoder
            cur, tgt = cur_nchannels['encoder'][i][1], tgt_nchannels['encoder'][i][1]
            assert cur >= tgt, "Cannot prune to more channels than exist."
            next_conv = model.encoder[i][2]
            prev_conv = model.encoder[i][0]
            keep_channels = importance_selection([prev_conv], [next_conv], tgt, args)

            prev_conv.weight.set_(prev_conv.weight.detach()[keep_channels])
            prev_conv.bias.set_(prev_conv.bias.detach()[keep_channels])
            prev_conv.out_channels = tgt
            next_conv.weight.set_(next_conv.weight.detach()[:, keep_channels])
            next_conv.in_channels = tgt

            # prune middle connection in decoder
            cur, tgt = cur_nchannels['decoder'][i], tgt_nchannels['decoder'][i]
            assert cur >= tgt, "Cannot prune to more channels than exist."
            next_conv = model.decoder[i][2]
            prev_conv = model.decoder[i][0]
            keep_channels = importance_selection([prev_conv], [next_conv], tgt, args)

            # first conv in decoder is glu layer too
            duplicated_channels = torch.cat([keep_channels, keep_channels+cur])
            prev_conv.weight.set_(prev_conv.weight.detach()[duplicated_channels])
            prev_conv.bias.set_(prev_conv.bias.detach()[duplicated_channels])
            prev_conv.out_channels = tgt*2
            # last conv in decoder is transpose
            next_conv.weight.set_(next_conv.weight.detach()[keep_channels])
            next_conv.in_channels = tgt        

    # print("Pruning complete.")

def sensitivity_scan(model, args):
    eval_args = eval_parser.parse_args(['--data_dir', 'egs/valentini', '--device', 'cuda', '--convert'])
    cur_nchannels = get_nchannels(model)
    sensitivities = {}
    for i in range(len(model.encoder)):
        for p in tqdm(torch.arange(0.2, 0.9, 0.2)):
            to_scan = []
            if i != 0 and args.scan_skips:
                to_scan.append(f'skip')
            if args.scan_encoder:
                to_scan.append(f'encoder')
            if args.scan_decoder:
                to_scan.append(f'decoder')

            for scan in to_scan:
                print(f'Scanning {scan} at {p}, layer {i}')
                tgt_nchannels = copy.deepcopy(cur_nchannels)
                newmodel = copy.deepcopy(model).cuda().eval()
                if scan == 'skip':
                    tgt_nchannels['encoder'][i][0] = int(p * cur_nchannels['encoder'][i][0])
                if scan == 'encoder':
                    tgt_nchannels['encoder'][i][1] = int(p * cur_nchannels['encoder'][i][1])
                if scan == 'decoder':
                    tgt_nchannels['decoder'][i] = int(p * cur_nchannels['decoder'][i])

                prune(newmodel, tgt_nchannels, args)
                pesq, stoi = evaluate(eval_args, model=newmodel)
                sensitivities.setdefault(f'{scan}_{i}_pesq', []).append(pesq)
                sensitivities.setdefault(f'{scan}_{i}_stoi', []).append(stoi)

    print('SENSITIVITIES: ', sensitivities)
    return sensitivities

def plot_sensitivity(sensitivities, baseline, args):
    fig, axs = plt.subplots(2, 3, figsize=(12,10))
    for k, v in sensitivities.items():
        row = axs[0] if 'pesq' in k else axs[1]
        ax = row[0] if 'skip' in k else row[1] if 'encoder' in k else row[2] if 'decoder' in k else row[1] if 'method 2' in k else row[0]
        ax.plot(torch.arange(20, 90, 20), list(reversed(v)), label=k)
        if '_1_' in k:
            ax.axhline(baseline[0 if 'pesq' in k else 1], linestyle='--', label='Baseline')
            ax.set_xlabel('Pruning %')
            ax.set_ylabel('PESQ' if 'pesq' in k else 'STOI')
            ax.set_title('Method 1' if 'method 1' in k else 'Method 2' if 'method 2' in k else 'Skip' if 'skip' in k else 'Encoder' if 'encoder' in k else 'Decoder')
        ax.legend()
    plt.savefig(args.output)

def plot_tradeoff(tradeoff, args):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    for k, v in tradeoff.items():
        if k=='designed+finetune' and not args.include_designed:
            continue
        pesqs = [d['pesq'] for d in v]
        stois = [d['stoi'] for d in v]
        sizes = [d['nparams']/(2**18) for d in v]

        axs[0].plot(sizes, pesqs, marker='x', label=k)
        axs[1].plot(sizes, stois, marker='x', label=k)
    axs[0].set_xlabel('Sizes (Mb)')
    axs[1].set_xlabel('Sizes (Mb)')
    axs[0].set_ylabel('PESQ')
    axs[1].set_ylabel('STOI')
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('Tradeoff accuracy vs size')
    plt.savefig(args.output)
        

def main(model=None):
    args = parser.parse_args()

    # Load model
    if not model:
        model = get_model(args).cpu()
    model.eval()

    # Uniform pruning
    if args.uniform:
        cur_nchannels = get_nchannels(model)
        tgt_nchannels = scale_down(cur_nchannels, args.p)
        prune(model, tgt_nchannels, args)
        model._init_args_kwargs[1]['prune_ratio'] = tgt_nchannels
        # Save model
        torch.save(serialize_model(model), args.output)

    if args.target:
        tgt_nchannels = json.loads(args.target)
        prune(model, tgt_nchannels, args)
        model._init_args_kwargs[1]['prune_ratio'] = tgt_nchannels
        torch.save(serialize_model(model), args.output)        
    
    # Sensitivity scan
    if args.sensitivity:
        sensitivity_scan(model, args)

    if args.plot_sensitivity:
        with open(args.sensitivity_path, 'r') as f:
            sensitivities = json.load(f)
        baseline = [2.514, 0.93]
        plot_sensitivity(sensitivities, baseline, args)

    if args.plot_tradeoff:
        with open(args.tradeoff_path, 'r') as f:
            tradeoff = json.load(f)
        plot_tradeoff(tradeoff, args)


if __name__ == "__main__":
    main()





