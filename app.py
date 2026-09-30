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

from core.soulx_preprocess import new_run_dir, preprocess_audio
from core.lyric_replace import (
    load_metadata,
    replace_lyrics,
    count_clean_notes,
    clean_text,
    save_metadata,
)
from core.soulx_engine import SoulXSingerEngine

SOULX = os.path.join(BASE, "SoulX-Singer")

# 模型引擎是全进程共享的单例；每个用户的识别结果放在 gr.State 里（按会话隔离），
# 否则多人/多标签页同时使用时会互相覆盖，B 点生成用的是 A 上传的音频。
_engine = None

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _char_count(text):
    return len(_CJK.findall(text or ""))


def _get_engine():
    global _engine
    if _engine is None:
        _engine = SoulXSingerEngine(
            model_path=os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model.pt"),
            config_path=os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml"),
            phoneset_path=os.path.join(SOULX, "soulxsinger", "utils", "phoneme", "phone_set.json"),
        )
    return _engine


def recognize(audio_path, progress=gr.Progress()):
    """上传干声后：自动识别原歌词 + 音符转写。返回值最后一项写入会话 state。"""
    if not audio_path:  # 清空音频
        return "", "尚未识别", None
    progress(0.1, desc="正在识别歌词与音符（约 30-60s）...")
    workdir = new_run_dir(os.path.join(BASE, "outputs"), audio_path)
    metadata_path, vocal_path = preprocess_audio(
        audio_path, save_dir=workdir, language="Mandarin", vocal_sep=False
    )
    metadata = load_metadata(metadata_path)
    total, per_phrase = count_clean_notes(metadata)
    session = {"metadata": metadata, "prompt_wav": vocal_path, "workdir": workdir}
    orig = clean_text(metadata)
    progress(1.0, desc="识别完成")
    return orig, f"原词 {total} 字，共 {len(per_phrase)} 句（每行一句）", session


def live_count(new_lyrics, session):
    total, per_phrase = count_clean_notes(session["metadata"]) if session else (0, [])
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


def generate(new_lyrics, control_mode, session, progress=gr.Progress()):
    """歌词替换 + 歌声合成。"""
    if not session:
        raise gr.Error("请先上传并识别原曲干声")

    progress(0.1, desc="歌词替换...")
    try:
        new_metadata = replace_lyrics(session["metadata"], new_lyrics, language="Mandarin")
    except ValueError as e:
        raise gr.Error(str(e))
    save_metadata(new_metadata, os.path.join(session["workdir"], "edit_metadata.json"))

    progress(0.3, desc="歌声合成（首次需加载模型）...")
    engine = _get_engine()
    audio, sr = engine.synthesize(
        prompt_wav_path=session["prompt_wav"],
        prompt_metadata=session["metadata"],
        target_metadata=new_metadata,
        control=control_mode,
        auto_shift=False,
        pitch_shift=0,
    )
    # 同一会话可多次生成，每次单独文件，避免覆盖正在播放/下载的上一版
    n = len([f for f in os.listdir(session["workdir"]) if f.startswith("cover")])
    out_path = os.path.join(session["workdir"], f"cover_{n + 1}.wav")
    sf.write(out_path, audio, sr)
    progress(1.0, desc="完成")
    return out_path


with gr.Blocks(title="改词翻唱 SoulX-Singer") as demo:
    session = gr.State(None)
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
            choices=[("score", "score"), ("melody", "melody")],
            value="score",
            label="音高控制模式",
            info="score（推荐）：跟量化音符，新词咬字准确；melody：跟原唱 F0 曲线，滑音/颤音更还原，但原唱咬字会带进来，可能串原词或含糊",
        )

    with gr.Row():
        audio_out = gr.Audio(label="翻唱结果（新干声）", type="filepath")

    audio_in.change(
        recognize, inputs=audio_in, outputs=[orig_lyrics, orig_info, session]
    ).then(live_count, inputs=[new_lyrics, session], outputs=count_info)
    new_lyrics.change(live_count, inputs=[new_lyrics, session], outputs=count_info)
    gen_btn.click(generate, inputs=[new_lyrics, control_mode, session], outputs=audio_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
