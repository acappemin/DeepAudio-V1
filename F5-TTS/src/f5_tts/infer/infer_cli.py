import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5-TTS | E2-TTS",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
    default="/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_vocos_l44k/pretrained_model_1200000.pt",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
)
parser.add_argument(
    "--end",
    type=int,
    default=99999999,
)
parser.add_argument(
    "--v2a_path",
    type=str,
    default="",
)
parser.add_argument(
    "--infer_list",
    type=str,
    default="/ailab-train/speech/zhanghaomin/datas/v2cdata/test.scp",
)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5-TTS")
model_cfg = args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml")))
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path)


# load TTS model

if model == "F5-TTS":
    model_cls = DiT
    model_cfg = OmegaConf.load(model_cfg).model.arch
    if not ckpt_file:  # path not specified, download from repo
        if vocoder_name == "vocos":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
        elif vocoder_name == "bigvgan":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))

elif model == "E2-TTS":
    assert args.model_cfg is None, "E2-TTS does not support custom model_cfg yet"
    assert vocoder_name == "vocos", "E2-TTS only supports vocoder vocos yet"
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if not ckpt_file:  # path not specified, download from repo
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)


# inference process


def main(ref_audio, ref_text, gen_text, energy):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            energy=energy,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments)-1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        return final_wave, final_sample_rate

        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

        #with open(wave_path, "wb") as f:
        #    sf.write(f.name, final_wave, final_sample_rate)
        #    # Remove silence
        #    if remove_silence:
        #        remove_silence_for_generated_wav(f.name)
        #    print(f.name)


import json
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


si_sdr = ScaleInvariantSignalDistortionRatio()


#def normalize_wav(waveform):
#    waveform = waveform - torch.mean(waveform)
#    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
#    return waveform * 0.5

def normalize_wav(waveform, waveform_ref):
    waveform = waveform / (torch.max(torch.abs(waveform))) * (torch.max(torch.abs(waveform_ref)))
    return waveform


if __name__ == "__main__":
    #scp1 = "/ailab-train/speech/zhanghaomin/datas/v2cdata/train.scp"
    #scp2 = "/ailab-train/speech/zhanghaomin/datas/v2cdata/test.scp"
    scp2 = args.infer_list

    #v2a_path = "/ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_neg/"
    v2a_path = args.v2a_path

    #with open(scp1, "r") as fr:
    #    lines1 = fr.readlines()
    with open(scp2, "r") as fr:
        lines2 = fr.readlines()
    #lines = lines1 + lines2
    lines = lines2

    datas = {}
    for line in lines:
        video, txt, wav = line.strip().split("\t")
        v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
        if not os.path.exists(video) or not os.path.exists(wav) or not os.path.exists(v2a_audio):
            continue
        spk = wav.rsplit("/", 1)[0]
        if spk not in datas:
            datas[spk] = []
        datas[spk].append([video, txt, wav])

    datas2 = []
    for spk in datas:
        for i in range(len(datas[spk])):
            p = -1
            for j in range(len(datas[spk])):
                if j == i:
                    continue
                if p == -1 or len(datas[spk][j][1]) > len(datas[spk][p][1]):
                    p = j
            datas2.append([datas[spk][i], datas[spk][p]])

    texts = []
    cond_lens = []
    prompts = []
    waveforms = []
    infos = []

    print("datas2", len(datas2))
    if False:
        with open("/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/data/v2c_test.lst", "w") as fw:
            for i, (data, data_p) in enumerate(datas2):
                video, txt, wav = data
                video_p, txt_p, wav_p = data_p

                v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
                v2a_audio_p = v2a_path + video_p.replace("/", "__") + ".flac"

                if not os.path.exists(video) or not os.path.exists(wav) or not os.path.exists(v2a_audio):
                    continue
                if not os.path.exists(video_p) or not os.path.exists(wav_p) or not os.path.exists(v2a_audio_p):
                    continue

                fw.write(wav_p+"\t"+video_p+"\t"+txt_p+"\t"+wav+"\t"+video+"\t"+txt+"\n")


    if False:
        sisdr_res = 0
        N = 0
        for i, (data, data_p) in enumerate(datas2):
            video, txt, wav = data
            video_p, txt_p, wav_p = data_p

            v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
            v2a_audio_p = v2a_path + video_p.replace("/", "__") + ".flac"

            if not os.path.exists(video) or not os.path.exists(wav) or not os.path.exists(v2a_audio):
                continue
            if not os.path.exists(video_p) or not os.path.exists(wav_p) or not os.path.exists(v2a_audio_p):
                continue

            wav_gen = "/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/outputs/gen/" + str(i).zfill(8) + ".wav"
            waveform_gen, sr_gen = torchaudio.load(wav_gen)
            duration_gen = waveform_gen.shape[-1] / sr_gen
            energy_gen = []
            for i in range(int(duration_gen/(256/24000))):
                energy_gen.append(waveform_gen[0,int(i*sr_gen*(256/24000)):int((i+1)*sr_gen*(256/24000))].abs().mean())
            energy_gen = np.array(energy_gen)
            energy_gen = energy_gen / max(energy_gen)

            energy = torch.from_numpy(np.load(wav+".npz")["arr_0"])
            #energy_pred = torch.from_numpy(np.load(v2a_audio+".npz")["arr_0"])
            energy_pred = torch.from_numpy(energy_gen)

            if energy_pred.shape[-1] < energy.shape[0]:
                energy_pred = torch.cat([energy_pred, torch.zeros(energy.shape[0]-energy_pred.shape[0])], dim=0)
            else:
                energy_pred = energy_pred[:energy.shape[0]]
            sisdr = si_sdr(energy_pred, energy)
            #print("sisdr", sisdr)
            sisdr_res += sisdr
            N += 1
        print("sisdr_res", N, sisdr_res/N)

    if True:
        for i, (data, data_p) in enumerate(datas2[args.start:args.end]):
            video, txt, wav = data
            video_p, txt_p, wav_p = data_p

            v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
            v2a_audio_p = v2a_path + video_p.replace("/", "__") + ".flac"

            if not os.path.exists(video) or not os.path.exists(wav) or not os.path.exists(v2a_audio):
                continue
            if not os.path.exists(video_p) or not os.path.exists(wav_p) or not os.path.exists(v2a_audio_p):
                continue

            energy = torch.from_numpy(np.load(v2a_audio+".npz")["arr_0"]).unsqueeze(0).unsqueeze(2)
            energy_p = torch.from_numpy(np.load(v2a_audio_p+".npz")["arr_0"]).unsqueeze(0).unsqueeze(2)
            #print("energy shape", energy_p.shape, energy.shape)
            #energy = torch.cat([energy_p, energy], dim=1)

            try:
                ####wav_gen, sr_gen = main(wav_p, txt_p, txt, [torch.zeros_like(energy_p), torch.zeros_like(energy)])
                ####wav_gen, sr_gen = main(wav_p, txt_p, txt, None)
                ####wav_gen, sr_gen = main(wav, txt, txt, None)
                wav_gen, sr_gen = main(wav_p, txt_p, txt, [energy_p, energy])
                ####wav_gen, sr_gen = main(wav, txt, txt, [energy.clone(), energy])
                wav_gen = torch.from_numpy(wav_gen).unsqueeze(0)
                assert(sr_gen == 24000)
            except:
                print("error generation", i+args.start, txt_p, txt)
                wav_gen = torch.zeros(1, 24000)
                sr_gen = 24000

            waveform, sr = torchaudio.load(wav)
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
            waveform_p, sr = torchaudio.load(wav_p)
            if sr != 24000:
                waveform_p = torchaudio.functional.resample(waveform_p, orig_freq=sr, new_freq=24000)
            #print(wav_gen.shape, wav_gen.max(), waveform.max(), waveform_p.max())

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(output_dir+"/ref/"):
                os.makedirs(output_dir+"/ref/")
            if not os.path.exists(output_dir+"/gen/"):
                os.makedirs(output_dir+"/gen/")
            if not os.path.exists(output_dir+"/tgt/"):
                os.makedirs(output_dir+"/tgt/")
            
            torchaudio.save(output_dir+"/ref/"+str(i+args.start).zfill(8)+".wav", waveform_p[0:1,:], 24000)
            torchaudio.save(output_dir+"/gen/"+str(i+args.start).zfill(8)+".wav", normalize_wav(wav_gen[0:1,:], waveform_p[0:1,:]), 24000)
            torchaudio.save(output_dir+"/tgt/"+str(i+args.start).zfill(8)+".wav", waveform[0:1,:], 24000)

            if not os.path.exists(output_dir+"/ref_nonorm/"):
                os.makedirs(output_dir+"/ref_nonorm/")
            if not os.path.exists(output_dir+"/gen_nonorm/"):
                os.makedirs(output_dir+"/gen_nonorm/")
            if not os.path.exists(output_dir+"/tgt_nonorm/"):
                os.makedirs(output_dir+"/tgt_nonorm/")
            torchaudio.save(output_dir+"/ref_nonorm/"+str(i+args.start).zfill(8)+".wav", waveform_p[0:1,:], 24000)
            torchaudio.save(output_dir+"/gen_nonorm/"+str(i+args.start).zfill(8)+".wav", wav_gen[0:1,:], 24000)
            torchaudio.save(output_dir+"/tgt_nonorm/"+str(i+args.start).zfill(8)+".wav", waveform[0:1,:], 24000)


"""
--ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt
--ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/model_14272.pt

CUDA_VISIBLE_DEVICES=0 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 0 --end 338 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=1 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 338 --end 676 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=2 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 676 --end 1014 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=3 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 1014 --end 1352 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=4 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 1352 --end 1690 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=5 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 1690 --end 2028 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=6 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 2028 --end 2366 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &
CUDA_VISIBLE_DEVICES=7 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_s1/ --start 2366 --end 2704 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c/pretrained_model_1200000.pt &

CUDA_VISIBLE_DEVICES=0 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 0 --end 338 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=1 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 338 --end 676 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=2 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 676 --end 1014 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=3 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 1014 --end 1352 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=4 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 1352 --end 1690 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=5 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 1690 --end 2028 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=6 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 2028 --end 2366 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
CUDA_VISIBLE_DEVICES=7 nohup python src/f5_tts/infer/infer_cli.py --output_dir outputs_v2c_s44/ --start 2366 --end 2704 --ckpt_file /ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/ckpts/v2c_s44/model_14272.pt --v2a_path /ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_s44/ &
"""

