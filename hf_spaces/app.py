#!/usr/bin/env python3
"""NanoGPT inference app for HuggingFace Spaces using Gradio."""
import sys
import os
import torch
import tiktoken
import gradio as gr
from model import GPT

# Load model
print("Loading model...")
# Try multiple paths
ckpt_paths = [
    'checkpoints/finetune_genz_1000k_best.pt',
    'finetune_genz_1000k_best.pt',
    '/app/checkpoints/finetune_genz_1000k_best.pt',
    '/app/finetune_genz_1000k_best.pt'
]
ckpt_path = None
for path in ckpt_paths:
    if os.path.exists(path):
        ckpt_path = path
        print(f"Found checkpoint at: {path}")
        break

if not ckpt_path:
    raise FileNotFoundError(f"Checkpoint not found in any of: {ckpt_paths}")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = GPT(vocab_size=50257)
state = ckpt['model']
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
enc = tiktoken.encoding_for_model('gpt2')
print("✅ Model loaded")

def generate(prompt, max_tokens=60, temperature=0.9, top_p=0.6):
    """Generate text from prompt."""
    try:
        # Normalize all-caps input to lowercase
        if prompt.isupper() and len(prompt) > 1:
            prompt = prompt.lower()

        # Tokenize
        tokens = enc.encode(prompt)
        num_prompt_tokens = len(tokens)
        idx = torch.tensor(tokens).unsqueeze(0)

        # Generate
        with torch.no_grad():
            idx = model.generate(idx, max_new_tokens=max_tokens,
                                temperature=temperature, top_p=top_p)

        # Decode only the newly generated tokens
        generated_tokens = idx[0, num_prompt_tokens:].tolist()
        generated = enc.decode(generated_tokens)

        # Post-processing
        # 1. Remove <|endoftext|> tokens
        generated = generated.replace('<|endoftext|>', '')
        generated = generated.strip()

        # 2. Skip first sentence (often fragment)
        sentences = generated.split('. ')
        if len(sentences) > 1:
            generated = '. '.join(sentences[1:])

        # 3. Keep first ~180 characters and cut at natural boundary
        if len(generated) > 180:
            truncated = generated[:180]
            best_cut = None
            for punct in ['.', '!', '?']:
                last_punct = truncated.rfind(punct)
                if last_punct > 80:
                    best_cut = last_punct + 1
                    break

            if best_cut:
                generated = generated[:best_cut]
            else:
                for i in range(min(len(truncated), 200), 80, -1):
                    if i < len(generated) and generated[i] == ' ':
                        if i > 80 and (generated[i-1].isalnum() or ord(generated[i-1]) > 127):
                            generated = generated[:i].rstrip() + '.'
                            break
                else:
                    generated = truncated + '.'
        elif not generated.endswith('.') and not generated.endswith('!') and not generated.endswith('?'):
            generated += '.'

        # 4. Stop at newline
        if '\n' in generated:
            generated = generated.split('\n')[0]

        # 5. Remove line-level repetition
        generated = generated.strip()
        lines = generated.split('\n')
        seen = set()
        deduped = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                if line_stripped not in seen:
                    deduped.append(line)
                    seen.add(line_stripped)
            else:
                deduped.append(line)

        generated = '\n'.join(deduped).strip()

        return generated

    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="NanoGPT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NanoGPT Chat")
    gr.Markdown("Gen Z slang language model trained on 1B tokens")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Your message",
                placeholder="Type something...",
                lines=2
            )
            with gr.Row():
                temp_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.5,
                    value=0.9,
                    step=0.1
                )
                top_p_slider = gr.Slider(
                    label="Top P",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05
                )
            submit_btn = gr.Button("Generate", variant="primary")

        output = gr.Textbox(
            label="NanoGPT response",
            lines=4,
            interactive=False
        )

    submit_btn.click(
        generate,
        inputs=[prompt, temp_slider, top_p_slider],
        outputs=output
    )

    prompt.submit(
        generate,
        inputs=[prompt, temp_slider, top_p_slider],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)