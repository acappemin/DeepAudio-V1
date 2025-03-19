import os
import sys

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter

import numpy as np
import torch
import torchaudio


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # process raw data
    #executor = ProcessPoolExecutor(max_workers=max_workers)
    #futures = []
    #
    #for subset in tqdm(SUB_SET):
    #    dataset_path = Path(os.path.join(dataset_dir, subset))
    #    [
    #        futures.append(executor.submit(deal_with_audio_dir, audio_dir))
    #        for audio_dir in dataset_path.iterdir()
    #        if audio_dir.is_dir()
    #    ]
    #for future in tqdm(futures, total=len(futures)):
    #    sub_result, durations, vocab_set = future.result()
    #    result.extend(sub_result)
    #    duration_list.extend(durations)
    #    text_vocab_set.update(vocab_set)
    #executor.shutdown()

    train_scp = "/ailab-train/speech/zhanghaomin/datas/v2cdata/test.scp"
    
    v2a_path = "/ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/"
    #v2a_path = "/ailab-train/speech/zhanghaomin/codes3/v2a_v2cdata/"

    with open(train_scp, "r") as fr:
        for line in fr.readlines():
            video, txt, audio = line.strip().split("\t")
            ####v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
            v2a_audio = v2a_path + video.replace("/", "__")[:-4] + ".wav"

            if not os.path.exists(video) or not os.path.exists(audio) or not os.path.exists(v2a_audio):
                print(video, audio, v2a_audio)
                continue
            waveform, sr = torchaudio.load(audio)
            duration = waveform.shape[-1] / sr
            waveform_v2a, sr_v2a = torchaudio.load(v2a_audio)
            duration_v2a = waveform_v2a.shape[-1] / sr_v2a

            if duration_v2a >= duration:
                waveform_v2a = waveform_v2a[:, :int(sr_v2a*duration)]
            else:
                waveform_v2a = torch.cat([waveform_v2a, torch.zeros([waveform_v2a.shape[0], int(sr_v2a*duration)-waveform_v2a.shape[1]])], dim=1)
            duration_v2a = duration
            
            energy_v2a = []
            for i in range(int(duration_v2a/(256/24000))):
                energy_v2a.append(waveform_v2a[0,int(i*sr_v2a*(256/24000)):int((i+1)*sr_v2a*(256/24000))].abs().mean())
            energy_v2a = np.array(energy_v2a)
            energy_v2a = energy_v2a / max(energy_v2a)
            #print(len(energy_v2a), max(energy_v2a), min(energy_v2a), energy_v2a.mean())
            np.savez(v2a_audio+".npz", energy_v2a)
            
            energy = []
            for i in range(int(duration/(256/24000))):
                energy.append(waveform[0,int(i*sr*(256/24000)):int((i+1)*sr*(256/24000))].abs().mean())
            energy = np.array(energy)
            energy = energy / max(energy)
            #print(len(energy), max(energy), min(energy), energy.mean())
            np.savez(audio+".npz", energy)
            
            d = {}
            d["audio_path"] = audio
            d["text"] = txt
            d["duration"] = duration
            d["energy"] = v2a_audio+".npz"
            result.append(d)
            duration_list.append(duration)
            text_vocab_set.update(list(txt))


    print(len(result), result[:2])  # 354218 [{'audio_path': '/ailab-train/speech/zhanghaomin/datas/libritts/LibriTTS/train-clean-100/7635/105409/7635_105409_000088_000000.wav', 'text': '"There is no \'but.\' I said, do you remember?"', 'duration': 2.31}, {'audio_path': '/ailab-train/speech/zhanghaomin/datas/libritts/LibriTTS/train-clean-100/7635/105409/7635_105409_000061_000002.wav', 'text': 'They know it.', 'duration': 0.76}]
    print(len(duration_list), duration_list[:2])  # 354218 [2.31, 0.76]
    print(len(text_vocab_set))  # 78

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    #SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    #dataset_dir = "/ailab-train/speech/zhanghaomin/datas/libritts/LibriTTS"
    #dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    dataset_name = "v2c_s44_test_char"
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
