## Installation

### 1. Create a conda environment
conda create -n v2as python=3.10
conda activate v2as

### 2. F5-TTS base install
cd ./F5-TTS
pip install -e .

### 3. Additional requirements
pip install -r requirements.txt
conda install cudnn


## Inference

### 1. V2A inference
bash v2a.sh

### 2. V2S inference
bash v2s.sh


## Eval

### 1. Download evaluation model checkpoints
English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)

### 2. Metrics Evaluation
bash eval_v2c.sh

