import time
from pathlib import Path
from hubconf import wavlm_large
import numpy as np
import pandas as pd
import torch
import argparse
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from fastprogress.fastprogress import master_bar, progress_bar
from torch import Tensor
import sys


DOWNSAMPLE_FACTOR = 320

# Global caches for storing features and synthesized data
global features_cache
features_cache = {}
global synthesized_data_cache
synthesized_data_cache = {}

# Function to create DataFrame from LibriSpeech data
def create_librispeech_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    folders = ['train-clean-100', 'dev-clean']
    print(f"[LIBRISPEECH] Calculating folders {folders}")
    for f in folders:
        all_files.extend(list((root_path/f).rglob('**/*.flac')))
    speakers = ['ls-' + f.stem.split('-')[0] for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

# Main function
def main(args):
    device = torch.device(args.device)
    SYNTH_WEIGHTINGS = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=25).float().to(device)[:, None]
    MATCH_WEIGHTINGS = F.one_hot(torch.tensor(args.matching_layer), num_classes=25).float().to(device)[:, None]

    print(f"Matching weights: {MATCH_WEIGHTINGS.squeeze()}\nSynthesis weights: {SYNTH_WEIGHTINGS.squeeze()}")
    ls_df = create_librispeech_df(Path(args.librispeech_path))

    print(f"Loading wavlm.")
    wavlm = wavlm_large(pretrained=True, progress=True, device=args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extract_features(ls_df, wavlm, args.device, Path(args.librispeech_path), Path(args.out_path), SYNTH_WEIGHTINGS, MATCH_WEIGHTINGS)
    print("All done!", flush=True)


def path_to_pools(path: Path, wavlm: nn.Module(), match_weights: Tensor, synth_weights: Tensor, device):
    """Calculates the matching pool for a given path to an audio track"""

    utterances_from_same_speaker = sorted(list(path.parent.rglob('**/*.flac')))
    utterances_from_same_speaker.remove(path)
    matching_pool = []
    synth_pool = []
    for pth in utterances_from_same_speaker:
        if pth in features_cache and pth in synthesized_data_cache:
            matching_features = features_cache[pth].float() # (seq_len, dim)
            synth_features = synthesized_data_cache[pth].float() # (seq_len, dim)
        else:
            features = get_full_features(pth, wavlm, device)
            matching_features = ( features*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)
            synth_features = ( features*synth_weights[:, None] ).sum(dim=0) # (seq_len, dim)
            features_cache[pth] = matching_features.half().cpu()
            synthesized_data_cache[pth] = synth_features.half().cpu()

        matching_pool.append(matching_features.cpu())
        synth_pool.append(synth_features.cpu())
    matching_pool = torch.concat(matching_pool, dim=0)
    synth_pool = torch.concat(synth_pool, dim=0)
    return matching_pool, synth_pool # (N, dim)


@torch.inference_mode()
def get_full_features(path, wavlm, device):
    """Gets full features for a given path to an audio track"""

    x, sr = torchaudio.load(path)
    assert sr == 16000
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)

    # extracts representation of each layer
    wav_input_16khz = x.to(device)
    rep, layer_results = wavlm.extract_features(wav_input_16khz, output_layer=wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
    features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)

    return features


def compute_fast_cosine_dist(source_features, matching_pool):
    """Computes fast cosine distance between source features and matching pool"""

    source_norms = torch.norm(source_features, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_features[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists



@torch.inference_mode()
def extract_features(df: pd.DataFrame, wavlm: nn.Module, device, ls_path: Path, out_path: Path, synth_weights: Tensor, match_weights: Tensor):
    """Extracts features from data using the wavlm model"""
    
    pb = progress_bar(df.iterrows(), total=len(df))

    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (out_path/rel_path).with_suffix('.pt')
        if args.resume:
            if targ_path.is_file(): continue
        os.makedirs(targ_path.parent, exist_ok=True)

        if Path(row.path) in features_cache:
            source_features = features_cache[Path(row.path)].float()
        else:
            source_features = get_full_features(row.path, wavlm, device)
            source_features = ( source_features*match_weights[:, None] ).sum(dim=0) # (seq_len, dim)

        matching_pool, synth_pool = path_to_pools(row.path, wavlm, match_weights, synth_weights, device)

        if not args.prematch:
            out_features = source_features.cpu()
        else:
            dists = compute_fast_cosine_dist(source_features.cpu(), matching_pool.cpu()).cpu()
            best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
            out_features = synth_pool[best.indices].mean(dim=1) # (N, dim)

        if i < 3: print("Feature shape: ", out_features.shape, flush=True)
        torch.save(out_features.cpu().half(), str(targ_path))
        if hasattr(pb, 'child'):
            pb.child.comment = str(rel_path)
            pb.child.wait_for = min(pb.child.wait_for, 10)
            pb.main_bar.comment = str(rel_path)
        else:
            pb.wait_for = min(pb.wait_for, 10)
        pb.comment = str(rel_path)
        

        if i % 1000 == 0: 
            print(f"Done {i:,d}/{len(df):,d}", flush=True)
            features_cache.clear()
            synthesized_data_cache.clear()
            gc.collect()
            time.sleep(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computing matching features wavlm for librispeech dataset")

    parser.add_argument('--librispeech_path', required=True, type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--out_path', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--matching_layer', type=int, default=6)
    parser.add_argument('--synthesis_layer', type=int, default=6)
    parser.add_argument('--prematch', action='store_true', help='prematch')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    main(args)
