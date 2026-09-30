# -*- coding: utf-8 -*-
"""改词翻唱 WebUI（SoulX-Singer）：上传干声 + 输入新歌词 -> 一键生成新干声。"""
import os
import re
import sys

import gradio as gr
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from core.soulx_preprocess import REF_MAX_SECONDS, new_run_dir, preprocess_audio, preprocess_reference
from core.lyric_replace import (
    apply_structure,
    count_clean_notes,
    load_metadata,
    note_structure,
    parse_structure,
    replace_lyrics,
    save_metadata,
    structure_text,
)
from core.soulx_engine import SoulXSingerEngine, SoulXSingerSVCEngine, estimate_pitch_shift

SOULX = os.path.join(BASE, "SoulX-Singer")

# 模型引擎是全进程共享的单例；每个用户的识别结果放在 gr.State 里（按会话隔离），
# 否则多人/多标签页同时使用时会互相覆盖，B 点生成用的是 A 上传的音频。
_engine = None
_svc_engine = None

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


def _get_svc_engine():
    global _svc_engine
    if _svc_engine is None:
        _svc_engine = SoulXSingerSVCEngine()
    return _svc_engine


def _next_output(workdir, prefix):
    # 同一会话可多次生成，每次单独文件，避免覆盖正在播放/下载的上一版
    n = len([f for f in os.listdir(workdir) if f.startswith(prefix)])
    return os.path.join(workdir, f"{prefix}_{n + 1}.wav")


def recognize(audio_path, progress=gr.Progress()):
    """上传干声后：自动识别原歌词 + 音符转写。返回值最后一项写入会话 state。"""
    if not audio_path:  # 清空音频
        return "", "尚未识别", None
    progress(0.1, desc="正在识别歌词与音符（约 30-60s）...")
    workdir = new_run_dir(os.path.join(BASE, "outputs"), audio_path)
    res = preprocess_audio(audio_path, save_dir=workdir, language="Mandarin", vocal_sep=False)
    metadata = load_metadata(res.metadata_path)
    if not metadata:
        raise gr.Error("没有识别出演唱内容，请确认上传的是清晰的中文演唱干声")
    structure = note_structure(metadata)
    session = {
        "orig_metadata": metadata,  # 识别原样，校正总是基于它重新计算
        "metadata": metadata,  # 当前生效（可能已校正）的原词 metadata
        "structure": structure,
        "corrected": False,
        "vocal_wav": res.vocal_path,  # 原曲人声：默认音色 prompt，也是「只换音色」的转换对象
        "vocal_f0": res.f0_path,
        "workdir": workdir,
    }
    progress(1.0, desc="识别完成")
    return structure_text(structure), _structure_info(session), session


def recognize_ref(ref_path, progress=gr.Progress()):
    """上传参考音色：识别其歌词/音符（改词合成的 prompt 需要）并提取 F0。"""
    if not ref_path:
        return "未上传参考音色：使用原曲自身的音色", None
    progress(0.1, desc="正在分析参考音色（约 20-40s）...")
    workdir = new_run_dir(os.path.join(BASE, "outputs"), ref_path)
    res = preprocess_reference(ref_path, save_dir=workdir, language="Mandarin")
    metadata = load_metadata(res.metadata_path)
    f0 = np.load(res.f0_path)
    if not (f0 > 0).any():
        raise gr.Error("参考音频中没有检测到演唱，请换一段清晰的演唱干声")
    ref = {"metadata": metadata or None, "vocal_wav": res.vocal_path, "vocal_f0": res.f0_path}
    dur = sf.info(res.vocal_path).duration
    note = "" if metadata else "；未识别出歌词，只能用于「只换音色」"
    progress(1.0, desc="完成")
    return f"✅ 参考音色已就绪（使用前 {dur:.0f} 秒{note}）", ref


_AUTO_SHIFT_CHOICES = [("按八度（保持原调）", "octave"), ("按半音（贴合音域，会变调）", "semitone"), ("关闭", "off")]


def _pitch_shift(session, ref, auto_mode, manual):
    """返回 (总移调半音数, 说明文字)。自动部分只在有参考音色时计算，且全曲统一。"""
    manual = int(manual)
    auto = 0
    if ref and auto_mode != "off":
        auto = estimate_pitch_shift(np.load(ref["vocal_f0"]), np.load(session["vocal_f0"]), auto_mode)
    total = auto + manual
    parts = ([f"自动 {auto:+d}"] if ref and auto_mode != "off" else []) + ([f"手动 {manual:+d}"] if manual else [])
    if not parts:
        return 0, "未变调"
    return total, f"变调 {total:+d} 个半音（" + "，".join(parts) + "）"


def _structure_info(session, prefix=""):
    total, per_phrase = count_clean_notes(session["metadata"], structure=session["structure"])
    return f"{prefix}原词 {total} 字，共 {len(per_phrase)} 句（每行一句）"


def correct(text, session):
    """用户编辑原词结构：合法则立即生效（改字/改转音/改分句），否则提示错误并保留上一次有效结构。"""
    if not session:
        return "尚未识别", session
    try:
        structure = parse_structure(text)
        metadata = apply_structure(session["orig_metadata"], structure, language="Mandarin")
    except ValueError as e:
        return f"🔴 {e}（仍使用上一次有效的原词结构）", session
    changed = structure != note_structure(session["orig_metadata"])
    session = {**session, "metadata": metadata, "structure": structure, "corrected": changed}
    return _structure_info(session, "✅ 已应用校正：" if changed else ""), session


def reset_structure(session):
    """还原为自动识别的原词结构。"""
    if not session:
        return "", "尚未识别", session
    structure = note_structure(session["orig_metadata"])
    session = {**session, "metadata": session["orig_metadata"], "structure": structure, "corrected": False}
    return structure_text(structure), _structure_info(session), session


def live_count(new_lyrics, session):
    total, per_phrase = (
        count_clean_notes(session["metadata"], structure=session["structure"]) if session else (0, [])
    )
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


def generate(new_lyrics, control_mode, auto_shift, manual_shift, session, ref, progress=gr.Progress()):
    """歌词替换 + 歌声合成。有参考音色时用它的音色，否则用原曲自身的音色。"""
    if not session:
        raise gr.Error("请先上传并识别原曲干声")
    if ref and not ref["metadata"]:
        raise gr.Error("参考音频未识别出歌词，无法用于改词合成（「只换音色」仍可用）")

    progress(0.1, desc="歌词替换...")
    try:
        new_metadata = replace_lyrics(
            session["metadata"], new_lyrics, language="Mandarin", structure=session["structure"]
        )
    except ValueError as e:
        raise gr.Error(str(e))
    save_metadata(new_metadata, os.path.join(session["workdir"], "edit_metadata.json"))
    if session["corrected"]:
        save_metadata(session["metadata"], os.path.join(session["workdir"], "corrected_metadata.json"))
        # 分句信息不在 metadata 里，单独保存结构文本；可直接用于 CLI 的 --original-structure
        with open(os.path.join(session["workdir"], "original_structure.txt"), "w", encoding="utf-8") as f:
            f.write(structure_text(session["structure"]) + "\n")

    shift, shift_info = _pitch_shift(session, ref, auto_shift, manual_shift)
    progress(0.3, desc="歌声合成（首次需加载模型）...")
    engine = _get_engine()
    audio, sr = engine.synthesize(
        prompt_wav_path=ref["vocal_wav"] if ref else session["vocal_wav"],
        prompt_metadata=ref["metadata"] if ref else session["metadata"],
        target_metadata=new_metadata,
        control=control_mode,
        auto_shift=False,  # 上游逐段计算，改为全曲统一的 shift
        pitch_shift=shift,
    )
    out_path = _next_output(session["workdir"], "cover")
    sf.write(out_path, audio, sr)
    progress(1.0, desc="完成")
    timbre = "参考音色" if ref else "原曲音色"
    return out_path, f"改词翻唱完成：{timbre}，{shift_info}"


def convert_timbre(auto_shift, manual_shift, session, ref, progress=gr.Progress()):
    """只换音色（SVC）：原曲的歌词、旋律、节奏不变，音色换成参考音色。"""
    if not session:
        raise gr.Error("请先上传原曲干声")
    if not ref:
        raise gr.Error("「只换音色」需要先上传参考音色")
    shift, shift_info = _pitch_shift(session, ref, auto_shift, manual_shift)
    progress(0.2, desc="音色转换（首次需加载 SVC 模型）...")
    audio, sr = _get_svc_engine().convert(
        prompt_wav_path=ref["vocal_wav"],
        prompt_f0_path=ref["vocal_f0"],
        target_wav_path=session["vocal_wav"],
        target_f0_path=session["vocal_f0"],
        pitch_shift=shift,
    )
    out_path = _next_output(session["workdir"], "timbre")
    sf.write(out_path, audio, sr)
    progress(1.0, desc="完成")
    return out_path, f"只换音色完成：歌词不变，{shift_info}"


with gr.Blocks(title="改词翻唱 SoulX-Singer") as demo:
    session = gr.State(None)
    ref_state = gr.State(None)
    gr.Markdown(
        "# 改词翻唱（本地 SoulX-Singer）\n上传原曲干声，输入改后歌词，一键生成原旋律、原节奏的新干声；"
        "默认保留原曲音色，也可上传参考音色换成另一位歌手的声音。"
    )

    with gr.Row():
        audio_in = gr.Audio(label="① 原曲干声（仅中文）", type="filepath")
        with gr.Column():
            ref_in = gr.Audio(label=f"参考音色（可选）：另一位歌手的中文演唱干声，取前 {REF_MAX_SECONDS} 秒", type="filepath")
            ref_info = gr.Markdown("未上传参考音色：使用原曲自身的音色")

    with gr.Row():
        with gr.Column():
            orig_lyrics = gr.Textbox(
                label="识别出的原歌词（可直接校正）",
                info="每行一句；~ 表示转音（延续前一个字的拖音）。可改错字、把字改成 ~ 或把 ~ 改成字、移动换行来合并/拆分句子；字与 ~ 的总数不能变",
                lines=14,
            )
            with gr.Row():
                orig_info = gr.Markdown("尚未识别")
                reset_btn = gr.Button("还原识别结果", size="sm", scale=0)
        with gr.Column():
            new_lyrics = gr.Textbox(
                label="② 改后歌词（每行一句，转音自动补全）",
                info="每行字数与左侧对应行的字数一致（~ 不计）",
                lines=14,
            )
            count_info = gr.Markdown("当前 0 字")

    with gr.Row():
        gen_btn = gr.Button("生成翻唱", variant="primary")
        svc_btn = gr.Button("只换音色（不改词）")
    with gr.Row():
        control_mode = gr.Radio(
            choices=[("melody", "melody"), ("score", "score")],
            value="melody",
            label="音高控制模式",
            info="melody（默认）：跟原唱 F0 曲线，滑音/颤音/转音最还原；若出现串原词或咬字含糊，改用 score。score：跟量化音符，咬字准确但唱腔细节较少",
        )
        with gr.Column():
            auto_shift = gr.Radio(
                choices=_AUTO_SHIFT_CHOICES,
                value="octave",
                label="自动变调到参考音色的音域",
                info="仅在上传参考音色时生效，按两位歌手的音高中位数计算、全曲统一。按八度：保持原调，可配原曲伴奏；按半音：最贴合参考歌手的音域，但整首歌会变调",
            )
            manual_shift = gr.Slider(-12, 12, value=0, step=1, label="手动升降调（半音）", info="在自动变调基础上再调整；+12 为升一个八度")

    with gr.Row():
        audio_out = gr.Audio(label="结果（新干声）", type="filepath")
    out_info = gr.Markdown("")

    audio_in.change(
        recognize, inputs=audio_in, outputs=[orig_lyrics, orig_info, session]
    ).then(live_count, inputs=[new_lyrics, session], outputs=count_info)
    ref_in.change(recognize_ref, inputs=ref_in, outputs=[ref_info, ref_state])
    # .input 只响应用户编辑，识别结果写入文本框时不会触发
    orig_lyrics.input(
        correct, inputs=[orig_lyrics, session], outputs=[orig_info, session]
    ).then(live_count, inputs=[new_lyrics, session], outputs=count_info)
    reset_btn.click(
        reset_structure, inputs=session, outputs=[orig_lyrics, orig_info, session]
    ).then(live_count, inputs=[new_lyrics, session], outputs=count_info)
    new_lyrics.change(live_count, inputs=[new_lyrics, session], outputs=count_info)
    gen_btn.click(
        generate,
        inputs=[new_lyrics, control_mode, auto_shift, manual_shift, session, ref_state],
        outputs=[audio_out, out_info],
    )
    svc_btn.click(
        convert_timbre,
        inputs=[auto_shift, manual_shift, session, ref_state],
        outputs=[audio_out, out_info],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        # 结果写在 <项目>/outputs 下；Gradio 默认只允许返回当前目录和临时目录的文件，
        # 不在项目目录下启动（python /path/to/app.py）时生成会报 InvalidPathError
        allowed_paths=[os.path.join(BASE, "outputs")],
    )
