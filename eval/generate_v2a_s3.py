import os
import torch
import torchaudio


#def normalize_wav(waveform):
#    waveform = waveform - torch.mean(waveform)
#    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
#    return waveform * 0.5

def normalize_wav(waveform, waveform_ref):
    waveform = waveform / (torch.max(torch.abs(waveform))) * (torch.max(torch.abs(waveform_ref)))
    return waveform


#with open("/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/data/v2c_test.lst", "r") as fr:
with open("/ailab-train/speech/zhanghaomin/codes3/F5-TTS-main/data/v2c_test_s3.lst", "r") as fr:
    lines = fr.readlines()


v2a_path = "/ailab-train/speech/zhanghaomin/codes3/MMAudio-main/output_v2c_neg/"
output_dir = "outputs_v2a_s3/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir+"/ref/"):
    os.makedirs(output_dir+"/ref/")
if not os.path.exists(output_dir+"/gen/"):
    os.makedirs(output_dir+"/gen/")
if not os.path.exists(output_dir+"/tgt/"):
    os.makedirs(output_dir+"/tgt/")


for idx, line in enumerate(lines):
    wav_p, video_p, txt_p, wav, video, txt = line.strip().split("\t")

    v2a_audio = v2a_path + video.replace("/", "__") + ".flac"
    #v2a_audio_p = v2a_path + video_p.replace("/", "__") + ".flac"
    assert(video_p == "None")

    waveform, sr = torchaudio.load(wav)
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
    waveform_p, sr = torchaudio.load(wav_p)
    if sr != 24000:
        waveform_p = torchaudio.functional.resample(waveform_p, orig_freq=sr, new_freq=24000)
    waveform_v2a, sr = torchaudio.load(v2a_audio)
    if sr != 24000:
        waveform_v2a = torchaudio.functional.resample(waveform_v2a, orig_freq=sr, new_freq=24000)

    torchaudio.save(output_dir+"/ref/"+str(idx).zfill(8)+".wav", waveform_p[0:1,:], 24000)
    torchaudio.save(output_dir+"/gen/"+str(idx).zfill(8)+".wav", normalize_wav(waveform_v2a[0:1,:], waveform_p[0:1,:]), 24000)
    torchaudio.save(output_dir+"/tgt/"+str(idx).zfill(8)+".wav", waveform[0:1,:], 24000)

    if not os.path.exists(output_dir+"/ref_nonorm/"):
        os.makedirs(output_dir+"/ref_nonorm/")
    if not os.path.exists(output_dir+"/gen_nonorm/"):
        os.makedirs(output_dir+"/gen_nonorm/")
    if not os.path.exists(output_dir+"/tgt_nonorm/"):
        os.makedirs(output_dir+"/tgt_nonorm/")
    torchaudio.save(output_dir+"/ref_nonorm/"+str(idx).zfill(8)+".wav", waveform_p[0:1,:], 24000)
    torchaudio.save(output_dir+"/gen_nonorm/"+str(idx).zfill(8)+".wav", waveform_v2a[0:1,:], 24000)
    torchaudio.save(output_dir+"/tgt_nonorm/"+str(idx).zfill(8)+".wav", waveform[0:1,:], 24000)

