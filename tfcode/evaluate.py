import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
import torch
import numpy as np
import argparse
from tqdm import tqdm

from denoiser.evaluate import get_pesq, get_stoi
# from denoiser.dsp import pad, unpad
from denoiser.data import NoisyCleanSet

PAD_MIN, PAD_STRIDE = 597, 256

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the TFLite model")
parser.add_argument("--data_dir", type=str, required=True, help="directory including noisy.json and clean.json files")
parser.add_argument("--matching", type=str, default="sort", help="set this to dns for the dns dataset.")
parser.add_argument("--convert", action="store_true", help="Convert to 16kHz before evaluation")
parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
parser.add_argument("--compute_length", type=int, default=160085, help="Length of segment model can compute")
args = parser.parse_args()

def evaluate(model, loader):
    cnt, pesq, stoi = 0, 0, 0
    for noisy, clean in tqdm(loader):
        estimate = predict(model, noisy)
        clean = clean.cpu().numpy()[:, 0, :]
        estimate = estimate[:, 0, :]
        pesq_i = get_pesq(clean, estimate, sr=args.sample_rate)
        stoi_i = get_stoi(clean, estimate, sr=args.sample_rate)
        # print(f"PESQ: {pesq_i}, STOI: {stoi_i}")
        pesq += pesq_i
        stoi += stoi_i
        cnt += clean.shape[0]

    print(f"PESQ: {pesq / cnt}, STOI: {stoi / cnt}")
    return {'pesq': pesq / cnt, 'stoi': stoi / cnt}

def get_loader(args):
    dataset = NoisyCleanSet(args.data_dir, matching=args.matching, sample_rate=args.sample_rate, convert=args.convert)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    return data_loader

def get_interpreter(args):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    return interpreter

def predict(interpreter, noisy):
    print("input:", noisy.shape)
    output = np.hstack([predict_segment(interpreter, noisy[..., i* args.compute_length: (i+ 1) * args.compute_length]) for i in range(noisy.shape[1] // args.compute_length)])
    rest = np.hstack([noisy[..., - len(noisy) % args.compute_length: ] , np.zeros_like((1, args.compute_length))])
    rest = predict_segment(interpreter, rest)[..., :len(noisy) % args.compute_length]
    output =  np.hstack([output, rest])
    print("output:", output.shape)
    return output

def predict_segment(interpreter, noisy):
    """
    Noisy: B x T
    """
    # noisy, length = pad(noisy, PAD_MIN, PAD_STRIDE)

    # Get input and output tensors.
    print("segment input", noisy.shape)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = noisy
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
    # return unpad(output_data, length)

def main(args):
    interpreter = get_interpreter(args)
    loader = get_loader(args)
    evaluate(interpreter, loader)

if __name__ == "__main__":
    main(args)
