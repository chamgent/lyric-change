# -*- coding: utf-8 -*-
"""改词翻唱 WebUI（SoulX-Singer）：上传干声 + 输入新歌词 -> 一键生成新干声。"""
import os
import re
import sys

import gradio as gr
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from core.soulx_preprocess import preprocess_audio
from core.lyric_replace import (
    load_metadata,
    replace_lyrics,
    count_clean_notes,
    clean_text,
    save_metadata,
)
from core.soulx_engine import SoulXSingerEngine

SOULX = os.path.join(BASE, "SoulX-Singer")

_state = {
    "metadata": None,
    "audio_path": None,
    "workdir": None,
    "engine": None,
}

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _char_count(text):
    return len(_CJK.findall(text or ""))


def _get_engine():
    if _state["engine"] is None:
        _state["engine"] = SoulXSingerEngine(
            model_path=os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model.pt"),
            config_path=os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml"),
            phoneset_path=os.path.join(SOULX, "soulxsinger", "utils", "phoneme", "phone_set.json"),
        )
    return _state["engine"]


def recognize(audio_path, progress=gr.Progress()):
    """上传干声后：自动识别原歌词 + 音符转写。"""
    if not audio_path:
        raise gr.Error("请先上传原曲干声")
    progress(0.1, desc="正在识别歌词与音符（约 30-60s）...")
    name = os.path.splitext(os.path.basename(audio_path))[0]
    workdir = os.path.join(BASE, "outputs", name)
    os.makedirs(workdir, exist_ok=True)
    metadata_path = preprocess_audio(
        audio_path, save_dir=workdir, language="Mandarin", vocal_sep=False
    )
    metadata = load_metadata(metadata_path)
    total, per_phrase = count_clean_notes(metadata)
    _state["metadata"] = metadata
    _state["audio_path"] = audio_path
    _state["workdir"] = workdir
    orig = clean_text(metadata)
    progress(1.0, desc="识别完成")
    return orig, f"原词 {total} 字，共 {len(per_phrase)} 句（每行一句）"


def live_count(new_lyrics):
    total, per_phrase = count_clean_notes(_state["metadata"]) if _state["metadata"] else (0, [])
    if total == 0:
        return f"当前 {_char_count(new_lyrics)} 字"
    lines = [l.strip() for l in (new_lyrics or "").splitlines() if l.strip()]
    if len(lines) != len(per_phrase):
        return f"🔴 行数不对：需 {len(per_phrase)} 行（每行一句），当前 {len(lines)} 行"
    diffs = []
    for i, (line, need) in enumerate(zip(lines, per_phrase)):
        d = _char_count(line) - need
        if d != 0:
            diffs.append(f"第{i + 1}行{d:+d}字")
    if not diffs:
        return f"🟢 匹配（{len(per_phrase)} 句，共 {total} 字）"
    return "🔴 " + "、".join(diffs)


def generate(new_lyrics, control_mode, progress=gr.Progress()):
    """歌词替换 + 歌声合成。"""
    if not _state["metadata"]:
        raise gr.Error("请先上传并识别原曲干声")

    progress(0.1, desc="歌词替换...")
    try:
        new_metadata = replace_lyrics(_state["metadata"], new_lyrics, language="Mandarin")
    except ValueError as e:
        raise gr.Error(str(e))
    save_metadata(new_metadata, os.path.join(_state["workdir"], "edit_metadata.json"))

    progress(0.3, desc="歌声合成（首次需加载模型）...")
    engine = _get_engine()
    audio, sr = engine.synthesize(
        prompt_wav_path=_state["audio_path"],
        prompt_metadata=_state["metadata"],
        target_metadata=new_metadata,
        control=control_mode,
        auto_shift=False,
        pitch_shift=0,
    )
    out_path = os.path.join(_state["workdir"], "cover.wav")
    sf.write(out_path, audio, sr)
    progress(1.0, desc="完成")
    return out_path


with gr.Blocks(title="改词翻唱 SoulX-Singer") as demo:
    gr.Markdown("# 改词翻唱（本地 SoulX-Singer）\n上传原曲干声，输入改后歌词，一键生成原音色、原旋律、原节奏的新干声。")

    with gr.Row():
        audio_in = gr.Audio(label="① 原曲干声（仅中文）", type="filepath")

    with gr.Row():
        with gr.Column():
            orig_lyrics = gr.Textbox(label="识别出的原歌词（每行一句，供参考）", lines=14, interactive=False)
            orig_info = gr.Markdown("尚未识别")
        with gr.Column():
            new_lyrics = gr.Textbox(label="② 改后歌词（每行一句，转音自动补全）", lines=14)
            count_info = gr.Markdown("当前 0 字")

    with gr.Row():
        gen_btn = gr.Button("生成翻唱", variant="primary")
        control_mode = gr.Radio(
            choices=[("melody", "melody"), ("score", "score")],
            value="melody",
            label="音高控制模式",
            info="melody：跟真实唱腔 F0（滑音/颤音都保留，最还原）；score：跟量化音符（更干净，转音略淡）",
        )

    with gr.Row():
        audio_out = gr.Audio(label="翻唱结果（新干声）", type="filepath")

    audio_in.change(recognize, inputs=audio_in, outputs=[orig_lyrics, orig_info])
    new_lyrics.change(live_count, inputs=new_lyrics, outputs=count_info)
    gen_btn.click(generate, inputs=[new_lyrics, control_mode], outputs=audio_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
