import os

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


if __name__ == "__main__":
    from pathlib import Path

    import librosa
    from tqdm import tqdm

    import sys

    test_lst = sys.argv[1]
    output_path = sys.argv[2]


    device = "cpu"
    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)

    ecos = 0
    nums = 0

    not_found = 0

    with open(test_lst, "r") as fr:
        lines = fr.readlines()

    path = output_path

    for idx, line in enumerate(lines):
        gen_wav = path + "gen/" + str(idx).zfill(8) + ".wav"
        target = path + "tgt/" + str(idx).zfill(8) + ".wav"

        if Path(gen_wav).exists() and Path(target).exists():
            try:
                wav = librosa.load(gen_wav, sr=16000)[0]
            except Exception as e:
                print(f"Error in {gen_wav}, {e}")
                not_found += 1
                continue
            try:
                target = librosa.load(target, sr=16000)[0]
            except Exception as e:
                not_found += 1
                print(f"Error in {target}, {e}")
                continue

            with torch.inference_mode():
                gen_emo_embs = process_func(wav, 16000, embeddings=True)
                target_emo_embs = process_func(target, 16000, embeddings=True)

            emo_cos = np.sum(gen_emo_embs * target_emo_embs) / (
                np.linalg.norm(gen_emo_embs) * np.linalg.norm(target_emo_embs)
            )
            emo_acc = emo_cos * 100
        else:
            # raise FileNotFoundError(wav, target)
            not_found += 1
            continue

        _cos = emo_acc

        ecos += _cos
        nums += 1

    print(f"EMO_SIM: {ecos / nums:.3f}")

