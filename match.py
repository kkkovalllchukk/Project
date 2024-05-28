from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor
from wavlm.WavLM import WavLM
from utils import generate_matrix_from_index
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

SPEAKER_LAYER = 6  # Layer that contains speaker information
SPEAKER_WEIGHTS = generate_matrix_from_index(SPEAKER_LAYER)  # Weights for the layer with speaker information

def compute_cosine_dist(source_features: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """
    Computes the cosine distance between the source features and the matching pool
    source_features - is a tensor representing the source features.
    matching_pool - is a tensor representing the matching pool.
    device - is a parameter that determines on which device the computations will be performed
    """
    source_norms = torch.norm(source_features, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_features[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists

class VoiceChanger(nn.Module):
    """ Class for using the k-nearest neighbors method for voice conversion (VC)"""

    def __init__(self,
        wavlm: WavLM,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device='cuda'
    ) -> None:
        """ Initialization of kNN-VC matcher. 
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan configuration to use when vocoding.
        """
        super().__init__()
        # sets which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_WEIGHTS, device=device)[:, None]
        # loads hifigan
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        # stores wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate
        self.hop_length = 320

    def get_matching_set(self, wavs: list[Path] | list[Tensor], weights=None, vad_trigger_level=7) -> Tensor:
        """ Gets concatenated wavlm features for a matching set, using all audio tracks in `wavs`, 
        specified as a list of paths or a list of loaded tensors of audio tracks of shape (channels, T), assuming a sampling rate of 16 kHz.
        Optionally specify custom weighting of WavLM features with `weights`.
        """
        features = []
        for p in wavs:
            features.append(self.get_features(p, weights=self.weighting if weights is None else weights, vad_trigger_level=vad_trigger_level))
        
        features = torch.concat(features, dim=0).cpu()
        return features
        

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """ Vocode features using hifigan. `c` has shape (bs, seq_len, c_dim) """
        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat

    @torch.inference_mode()
    def get_features(self, path, weights=None, vad_trigger_level=0):
        """Returns features of `path` audio track as a tensor of shape (seq_len, dim), 
        optionally performs VAD trimming at the start/end with `vad_trigger_level`.
        """
        # loads audio
        if weights == None: weights = self.weighting
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]
                
        if not sr == self.sr :
            print(f"resampling {sr} to {self.sr} in {path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
            
        # trims silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            x = waveform_end_trim

        # extracts representation of each layer
        wav_input_16khz = x.to(self.device)
        if torch.allclose(weights, self.weighting):
            # uses fast path
            features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_LAYER, ret_layer_results=False)[0]
            features = features.squeeze(0)
        else:
            # uses slower weighting
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # stores full sequence
            features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        
        return features



    @torch.inference_mode()
    def match(self, query_seq: Tensor, matching_set: Tensor, synth_set: Tensor = None, 
            topk: int = 4, tgt_loudness_db: float | None = -16,
            target_duration: float | None = None, device: str | None = None) -> Tensor:
        """ Performs kNN regression matching for `query_seq`, `matching_set` and `synth_set` tensors of shape (N, dim) with k=`topk`.
        Inputs:
            - `query_seq`: Tensor (N1, dim) of input/source query features.
            - `matching_set`: Tensor (N2, dim) of matching set used as 'training set' for the kNN algorithm.
            - `synth_set`: optional tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query vector to a vector in the matching set, then use the corresponding vector from the synth set during HiFiGAN synthesis. By default, and for best performance, this should be identical to the matching set. 
            - `topk`: k in kNN -- number of nearest neighbors to average.
            - `tgt_loudness_db`: decibels used to normalize the loudness of the output signal. Set to None to disable. 
            - `target_duration`: if a float value is set, interpolates the duration of the resulting audio track to be exactly this value in seconds.
            - `device`: if None, uses the default device at initialization. Otherwise uses the specified device
        Returns:
            - transformed audio track of shape (T,)
        """
        device = torch.device(device) if device is not None else self.device
        if synth_set is None: synth_set = matching_set.to(device)
        else: synth_set = synth_set.to(device)
        matching_set = matching_set.to(device)
        query_seq = query_seq.to(device)

        if target_duration is not None:
            target_samples = int(target_duration*self.sr)
            scale_factor = (target_samples/self.hop_length) / query_seq.shape[0] # n_targ_feats / n_input_feats
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

        dists = compute_cosine_dist(query_seq, matching_set, device=device)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)
        
        prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()
        
        # normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
        else: pred_wav = prediction
        return pred_wav
