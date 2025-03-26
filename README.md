<div align="center">
<p align="center">
  <h2>DeepAudio-V1</h2>
  <a href="https://arxiv.org">Paper</a> | <a href="https://acappemin.github.io/DeepAudio-V1.github.io">Webpage</a> | <a href="https://huggingface.co/spaces/lshzhm/DeepAudio-V1">Huggingface Demo</a> | <a href="https://huggingface.co/lshzhm/DeepAudio-V1">Models</a>
</p>
</div>


## DeepAudio-V1:Towards Multi-Modal Multi-Stage End-to-End Video to Speech and Audio Generation


## Results

<div style="padding: 0 0; text-align: center; display: flex; justify-content: space-around;">
	<p style="text-align: center;">Ground Truth</p>
	<p style="text-align: center;">Generated Audios</p>
</div>
<div class="row">
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000001.gt.mp4">
	</video>
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000001.gen.mp4">
	</video>
</div>
<div class="row">
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000003.gt.mp4">
	</video>
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000003.gen.mp4">
	</video>
</div>
<div class="row">
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000024.gt.mp4">
	</video>
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000024.gen.mp4">
	</video>
</div>
<div class="row">
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000025.gt.mp4">
	</video>
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000025.gen.mp4">
	</video>
</div>
<div class="row">
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000111.gt.mp4">
	</video>
	<video style="width: 49%;" controls>
		<source src="https://acappemin.github.io/DeepAudio-V1.github.io/assets/videos/00000111.gen.mp4">
	</video>
</div>



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

The models are available at https://huggingface.co/lshzhm/DeepAudio-V1. See [MODELS.md](./MODELS.md) for more details.

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

