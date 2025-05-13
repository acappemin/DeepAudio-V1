<div align="center">
<p align="center">
  <h2>DeepAudio-V1</h2>
  <a href="https://arxiv.org/abs/2503.22265">Paper</a> | <a href="https://acappemin.github.io/DeepAudio-V1.github.io">Webpage</a> | <a href="https://huggingface.co/lshzhm/DeepAudio-V1/tree/main">Models</a> | <a href="https://huggingface.co/spaces/lshzhm/DeepAudio-V1">Huggingface Demo</a> | <a href="https://colab.research.google.com/drive/1tfpv6DxD5q2EFUAIkQKfTI6fgOvWGTPy">Colab Demo</a> | <a href="https://replicate.com/acappemin/deepaudio-v1">Replicate Demo</a>
</p>
</div>


## DeepAudio-V1:Towards Multi-Modal Multi-Stage End-to-End Video to Speech and Audio Generation

[Haomin Zhang](https://scholar.google.com/citations?user=cxj9ZbAAAAAJ), [Chang Liu](), [Junjie Zheng](https://scholar.google.com/citations?hl=en&user=9lzy9McAAAAJ), [Zihao Chen](), [Chaofan Ding](), [Xinhan Di]()

AI Lab Giant Network, and University of Trento


## Results

**Objective metrics**

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/4afe5527-5a3e-40a2-9fe6-139db39144d7"/>

**Videos from unseen movies**

https://github.com/user-attachments/assets/5a226233-46a4-4746-b158-3e0ac31f192d

https://github.com/user-attachments/assets/922bbe72-6d99-422b-9c19-88aa31e213a4

https://github.com/user-attachments/assets/28a3f64c-2613-4e32-bb3d-d7331cf49a2a

https://github.com/user-attachments/assets/6221e347-68b6-4498-84fd-5a269d5ab94e

https://github.com/user-attachments/assets/91e89974-37a5-41fe-9e27-3f62fcd17360

https://github.com/user-attachments/assets/69719608-a438-485a-8681-5597654249be

https://github.com/user-attachments/assets/58b8903f-3e07-43ae-806f-076c505c0086

https://github.com/user-attachments/assets/b8631575-28a2-4446-b1f7-7b1e13bb67e2

https://github.com/user-attachments/assets/b4f4b7de-63be-4ee5-9b1f-05fe28365965

**Videos from V2C test set**

https://github.com/user-attachments/assets/2c474edd-e12a-41c4-970d-66e414ccf317

https://github.com/user-attachments/assets/160bf9e6-cbf9-4fa4-92b7-f303ea81cd3e

**For more results, please visit https://acappemin.github.io/DeepAudio-V1.github.io.**

## Installation

**1. Create a conda environment**

```bash
conda create -n v2as python=3.10
conda activate v2as
```

**2. F5-TTS base install**

```bash
cd ./F5-TTS
pip install -e .
```

**3. Additional requirements**

```bash
pip install -r requirements.txt
conda install cudnn
```

**Pretrained models**

The models are available at https://huggingface.co/lshzhm/DeepAudio-V1/tree/main. See [MODELS.md](./MODELS.md) for more details.

## Inference

**1. V2A inference**

```bash
bash v2a.sh
```

**2. V2S inference**

```bash
bash v2s.sh
```

**3. TTS inference**

```bash
bash tts.sh
```

## Evaluation

```bash
bash eval_v2c.sh
```

## Acknowledgement

- [MMAudio](https://github.com/hkchengrex/MMAudio) for video-to-audio backbone and pretrained models
- [F5-TTS](https://github.com/SWivid/F5-TTS) for text-to-speech and video-to-speech backbone
- [V2C](https://github.com/chenqi008/V2C) for animated movie benchmark
- [Wav2Vec2-Emotion](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) for emotion recognition in EMO-SIM evaluation.
- [WavLM-SV](https://huggingface.co/microsoft/wavlm-base-sv) for speech recognition in SPK-SIM evaluation.
- [Whisper](https://huggingface.co/Systran/faster-whisper-large-v3) for speech recognition in WER evaluation.

