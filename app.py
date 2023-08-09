import os
import json
import argparse
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import asyncio
import edge_tts
from datetime import datetime
from fairseq import checkpoint_utils
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from vc_infer_pipeline import VC
from config import (
    is_half,
    device
)
logging.getLogger("numba").setLevel(logging.WARNING)

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index, file_big_npy):
    def vc_fn(
        input_audio,
        f0_up_key,
        f0_method,
        index_rate
    ):
        try:
            if args.files:
                audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            else:
                if input_audio is None:
                    return "You need to upload an audio", None
                sampling_rate, audio = input_audio
                duration = audio.shape[0] / sampling_rate
                if duration > 10000000:
                    return "no", None
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio.transpose(1, 0))
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                file_big_npy,
                index_rate,
                if_f0,
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            )
            return "Success", (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)
    return vc_fn

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--files", action="store_true", default=False, help="load audio from path")
    args, unknown = parser.parse_known_args()
    load_hubert()
    models = []
    with open("weights/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)
    for name, info in models_info.items():
        if not info['enable']:
            continue
        title = info['title']
        cover = f"weights/{name}/{info['cover']}"
        index = f"weights/{name}/{info['feature_retrieval_library']}"
        npy = f"weights/{name}/{info['feature_file']}"
        cpt = torch.load(f"weights/{name}/{name}.pth", map_location="cpu")
        tgt_sr = cpt["config"][-1]
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(device)
        if is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, device, is_half)
        models.append((name, title, cover, create_vc_fn(tgt_sr, net_g, vc, if_f0, index, npy)))
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> RVC generator\n"
            "## <center> The input audio should be clean and pure voice without background music.\n"
            "[![buymeacoffee](https://badgen.net/badge/icon/buymeacoffee?icon=buymeacoffee&label)](https://www.buymeacoffee.com/spark808)\n\n"
        )
        with gr.Tabs():
            for (name, title, cover, vc_fn) in models:
                with gr.TabItem(name):
                    with gr.Row():
                        gr.Markdown(
                            '<div align="center">'
                            f'<div>{title}</div>\n'+
                            (f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else "")+
                            '</div>'
                        )
                    with gr.Row():
                        with gr.Column():
                            if args.files:
                                vc_input = gr.Textbox(label="Input audio path")
                            else:
                                vc_input = gr.Audio(label="Input audio")
                            vc_transpose = gr.Number(label="Transpose", value=0)
                            vc_f0method = gr.Radio(
                                label="Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies",
                                choices=["pm", "harvest"],
                                value="harvest",
                                interactive=True,
                            )
                            vc_index_ratio = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label="Retrieval feature ratio",
                                value=0.6,
                                interactive=True,
                            )
                            vc_submit = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            vc_output1 = gr.Textbox(label="Output Message")
                            vc_output2 = gr.Audio(label="Output Audio")
                vc_submit.click(vc_fn, [vc_input, vc_transpose, vc_f0method, vc_index_ratio], [vc_output1, vc_output2])
        app.queue(concurrency_count=1, max_size=20, api_open=args.api).launch(share=args.share)
