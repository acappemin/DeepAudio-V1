import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

from datetime import datetime

import traceback

import numpy as np
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


####CUDA_VISIBLE_DEVICES=0 python demo.py --output ./output
####CUDA_VISIBLE_DEVICES=4 nohup python demo.py --output ./output_v2c_neg --start 0 --end 1500 &


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k',
                        #default='small_16k',
                        #default='medium_44k',
                        #default='small_44k',
                        help='small_16k, small_44k, medium_44k, large_44k, large_44k_v2')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)
    
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=99999999)
    parser.add_argument('--scp', type=str, help='video list', default='/ailab-train/speech/zhanghaomin/datas/v2cdata/tmp.scp')
    parser.add_argument('--calc_energy', type=int, default=1)

    parser.add_argument('--mask_away_clip', action='store_true')

    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    if args.video:
        video_path: Path = Path(args.video).expanduser()
    else:
        video_path = None
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    print("full_precision", args.full_precision)
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    ####model.model_path = "/ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output/exp_1/exp_1_shadow.pth"
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    ####test_scp = "/ailab-train/speech/zhanghaomin/animation_dataset_v2a/test.scp"
    #test_scp = "/ailab-train/speech/zhanghaomin/datas/v2cdata/tmp.scp"
    #test_scp = "/ailab-train/speech/zhanghaomin/datas/v2cdata/test.scp"
    test_scp = args.scp
    
    lines = []
    with open(test_scp, "r") as fr:
        lines += fr.readlines()
    #with open(test_scp2, "r") as fr:
    #    lines += fr.readlines()
    tests = []
    for line in lines[args.start: args.end]:
        ####video_path, prompt = line.strip().split("\t")
        ####prompt = "the sound of " + prompt
        ####negative_prompt = ""
        video_path, _, audio_path = line.strip().split("\t")
        ####video_path = "/ailab-train/speech/zhanghaomin/datas/v2cdata/DragonII/DragonII_videos/Gobber/0725.mp4"
        prompt = ""
        #negative_prompt = "speech, voice, talking, speaking"
        negative_prompt = ""
        tests.append([video_path, prompt, negative_prompt, audio_path])
    
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "start")
    for video_path, prompt, negative_prompt, audio_path in tests:
        if video_path is not None:
            video_path = Path(video_path).expanduser()
            log.info(f'Using video {video_path}')
            try:
                video_info = load_video(video_path, args.duration)
            except:
                print("Error load_video", video_path)
                traceback.print_exc()
                continue
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            duration = video_info.duration_sec
            if mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            log.info('No video provided -- text-to-audio mode')
            clip_frames = sync_frames = None

        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        log.info(f'Prompt: {prompt}')
        log.info(f'Negative prompt: {negative_prompt}')

        audios = generate(clip_frames,
                          sync_frames, [prompt],
                          negative_text=[negative_prompt],
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]
        if video_path is not None:
            ####save_path = output_dir / f'{video_path.stem}.flac'
            save_path = str(output_dir) + "/" + str(video_path).replace("/", "__").strip(".") + ".flac"
        else:
            safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
            save_path = output_dir / f'{safe_filename}.flac'
        torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
        
        #### calculate energy
        if args.calc_energy:
            waveform_v2a, sr_v2a = torchaudio.load(save_path)
            duration_v2a = waveform_v2a.shape[-1] / sr_v2a

            if os.path.exists(audio_path):
                waveform, sr = torchaudio.load(audio_path)
                duration = waveform.shape[-1] / sr
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
            np.savez(save_path+".npz", energy_v2a)

        log.info(f'Audio saved to {save_path}')
        if video_path is not None and not skip_video_composite:
            ####video_save_path = output_dir / f'{video_path.stem}.mp4'
            video_save_path = str(output_dir) + "/" + str(video_path).replace("/", "__").strip(".") + ".mp4"
            make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
            log.info(f'Video saved to {video_save_path}')

        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "end")


if __name__ == '__main__':
    main()
