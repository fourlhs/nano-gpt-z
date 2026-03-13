---
title: NanoGPT
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# NanoGPT

A lightweight Gen Z slang language model powered by a finetuned GPT-2 architecture, trained on 1B tokens.

## Features

- **Fast inference**: ~3-5 seconds per generation on free GPU
- **Gen Z slang**: Trained specifically on internet slang and colloquial language
- **Simple interface**: Clean Gradio chat UI
- **Adjustable parameters**: Control temperature and top-p sampling

## Usage

Type any prompt and the model will generate a response in Gen Z slang style.

### Parameters
- **Temperature**: Controls randomness (0.1-1.5). Higher = more creative, lower = more deterministic
- **Top P**: Nucleus sampling threshold (0.1-1.0). Controls diversity of token selection

## Model Details

- Architecture: GPT-2 style transformer
- Vocabulary: 50,257 tokens (GPT-2 BPE encoding)
- Training: 1B tokens of Gen Z slang data
- Checkpoint: `finetune_genz_1000k_best.pt`

## How to Run Locally

```bash
# Install dependencies
pip install -r hf_spaces/requirements.txt

# Run the app
python hf_spaces/app.py
```

The app will be available at `http://localhost:7860`

## Deployment

This Space uses Docker for deployment on HuggingFace Spaces.