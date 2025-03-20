# Pretrained models

| Model    | Download link | File size |
| -------- | ------- | ------- |
| Speech synthesis model, based on MMAudio small 16kHz | <a href="https://huggingface.co" download="v2c_s16.pt">v2c_s16.pt</a> | 1.3G |
| Speech synthesis model, based on MMAudio small 44.1kHz | <a href="https://huggingface.co" download="v2c_s44.pt">v2c_s44.pt</a> | 1.3G |
| Speech synthesis model, based on MMAudio medium 44.1kHz | <a href="https://huggingface.co" download="v2c_m44.pt">v2c_m44.pt</a> | 1.3G |
| Speech synthesis model, based on MMAudio large 44.1kHz | <a href="https://huggingface.co" download="v2c_l44.pt">v2c_l44.pt</a> | 1.3G |
| MMAduio, small 16kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_16k.pth" download="mmaudio_small_16k.pth">mmaudio_small_16k.pth</a> | 601M |
| MMAduio, small 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_44k.pth" download="mmaudio_small_44k.pth">mmaudio_small_44k.pth</a> | 601M |
| MMAduio, medium 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_medium_44k.pth" download="mmaudio_medium_44k.pth">mmaudio_medium_44k.pth</a> | 2.4G |
| MMAduio, large 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k.pth" download="mmaudio_large_44k.pth">mmaudio_large_44k.pth</a> | 3.9G |
| MMAduio, large 44.1kHz, v2 | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth" download="mmaudio_large_44k_v2.pth">mmaudio_large_44k_v2.pth</a> | 3.9G |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| 44.1kHz VAE |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth">v1-44.pth</a> | 1.2G | 
| Synchformer visual encoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth">synchformer_state_dict.pth</a> | 907M |
| Whisper model for WER evaluation | <a href="https://huggingface.co/Systran/faster-whisper-large-v3" download="faster-whisper-large-v3">faster-whisper-large-v3</a> | 2.9G |
| WavLM model for SIM-O evaluation | <a href="https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view" download="wavlm_large_finetune.pth">wavlm_large_finetune.pth</a> | 1.2G |


The expected directory structure:

```bash
F5-TTS
├── ckpts
│   ├── v2c
│   │   ├── v2c_s16.pt
│   │   ├── v2c_s44.pt
│   │   ├── v2c_m44.pt
│   │   └── v2c_l44.pt
│   ├── faster-whisper-large-v3
│   └── wavlm_large_finetune.pth
└── ...
MMAudio
├── ext_weights
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   ├── v1-16.pth
│   └── v1-44.pth
├── weights
│   ├── mmaudio_small_16k.pth
│   ├── mmaudio_small_44k.pth
│   ├── mmaudio_medium_44k.pth
│   ├── mmaudio_large_44k.pth
│   └── mmaudio_large_44k_v2.pth
└── ...
```

