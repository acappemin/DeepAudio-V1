import os
from pathlib import Path

import librosa
import numpy as np
import torch

# from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

import sys

test_lst = sys.argv[1]
output_path = sys.argv[2]


# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv").cuda()

# the resulting embeddings can be used for cosine similarity-based retrieval
cosine_sim = torch.nn.CosineSimilarity(dim=-1)


with open(test_lst, "r") as fr:
    lines = fr.readlines()

path = output_path

scos = []

#for line in tqdm(val_list):
for idx, line in enumerate(lines):
    gen_wav = path + "gen/" + str(idx).zfill(8) + ".wav"
    target = path + "tgt/" + str(idx).zfill(8) + ".wav"

    if Path(gen_wav).exists() and Path(target).exists():
        try:
            wav = librosa.load(gen_wav, sr=16000)[0]
        except Exception as e:
            print(f"Error in {gen_wav}, {e}")
            continue
        try:
            target = librosa.load(target, sr=16000)[0]
        except Exception as e:
            print(f"Error in {target}, {e}")
            continue

        try:
            # audio files are decoded on the fly
            input1 = feature_extractor(wav, return_tensors="pt", sampling_rate=16000).to("cuda")
            embeddings1 = model(**input1).embeddings

            input2 = feature_extractor(target, return_tensors="pt", sampling_rate=16000).to("cuda")
            embeddings2 = model(**input2).embeddings

            similarity = cosine_sim(embeddings1[0], embeddings2[0])

        except Exception as e:
            print(f"Error in {gen_wav}, {e}")
            continue

        if 0 < similarity < 1:
            scos.append(similarity.detach().cpu().numpy())


print("SPK-SIM:", np.mean(scos), len(scos))

