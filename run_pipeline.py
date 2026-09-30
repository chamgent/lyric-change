# -*- coding: utf-8 -*-
"""端到端 CLI：干声 + 新歌词 -> 新干声。

用法:
    # 只识别：打印原词结构（每行一句，~ 表示转音），不合成
    python run_pipeline.py --audio 干声.wav
    # 改词合成
    python run_pipeline.py --audio 干声.wav --new-lyrics 新词.txt --output out.wav
    # 识别有误时：把打印的原词结构复制到文件里改正，再用 --original-structure 传入
    python run_pipeline.py --audio 干声.wav --original-structure 原词.txt --new-lyrics 新词.txt
    # 用另一位歌手的音色改词翻唱（默认自动变调到参考歌手的音域）
    python run_pipeline.py --audio 干声.wav --new-lyrics 新词.txt --ref-audio 参考.wav
    # 只换音色、不改词
    python run_pipeline.py --audio 干声.wav --ref-audio 参考.wav --svc --output out.wav

新词文件：每行一句（与原词句数/每句字数一致，转音自动补全）。
原词结构文件：可改错字、把字改成 ~ 或把 ~ 改成字、移动换行来合并/拆分句子；字与 ~ 的总数不能变。
"""
import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.soulx_preprocess import new_run_dir, preprocess_audio, preprocess_reference
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
from core.soulx_engine import AUTO_SHIFT_MODES, SoulXSingerEngine, SoulXSingerSVCEngine, estimate_pitch_shift

BASE = os.path.dirname(os.path.abspath(__file__))
SOULX = os.path.join(BASE, "SoulX-Singer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="原曲干声")
    ap.add_argument("--new-lyrics", help="新歌词文件(UTF-8)；省略则只识别并打印原词结构")
    ap.add_argument(
        "--original-structure",
        help="校正后的原词结构文件(UTF-8)，格式同运行时打印的「原词结构」",
    )
    ap.add_argument("--output", default="output_cover.wav")
    ap.add_argument("--language", default="Mandarin")
    ap.add_argument("--vocal-sep", action="store_true", help="输入含伴奏则开启人声分离")
    ap.add_argument(
        "--control",
        default="melody",
        choices=["melody", "score"],
        help="melody（默认）：跟原唱 F0 曲线，滑音/颤音/转音最还原；若出现串原词"
        "或咬字含糊，改用 score。score：跟量化音符，咬字准确但唱腔细节较少",
    )
    ap.add_argument("--ref-audio", help="参考音色：另一位歌手的中文演唱干声（取前 30 秒）；省略则用原曲自身的音色")
    ap.add_argument(
        "--auto-shift",
        default="octave",
        choices=AUTO_SHIFT_MODES,
        help="有参考音色时自动变调到参考歌手的音域（全曲统一）：octave（默认）只按整八度，保持原调、"
        "可配原曲伴奏；semitone 精确到半音，最贴合音域但整首歌会变调；off 关闭",
    )
    ap.add_argument("--pitch-shift", type=int, default=0, help="在自动变调基础上再升降的半音数")
    ap.add_argument("--svc", action="store_true", help="只换音色、不改词（需 --ref-audio，不需要 --new-lyrics）")
    args = ap.parse_args()

    if args.svc and not args.ref_audio:
        ap.error("--svc 需要同时指定 --ref-audio")
    if args.svc and (args.new_lyrics or args.original_structure):
        ap.error("--svc 只换音色、不改词，不能与 --new-lyrics / --original-structure 同时使用")

    args.audio = os.path.abspath(args.audio)
    args.output = os.path.abspath(args.output)
    if args.svc:
        run_svc(args)
        return

    new_lyrics = None
    if args.new_lyrics:
        with open(args.new_lyrics, encoding="utf-8") as f:
            new_lyrics = f.read()
    structure = None
    if args.original_structure:
        with open(args.original_structure, encoding="utf-8") as f:
            try:
                structure = parse_structure(f.read())
            except ValueError as e:
                sys.exit(f"原词结构文件有误：{e}")

    workdir = new_run_dir(os.path.join(BASE, "outputs"), args.audio)
    print(f"工作目录：{workdir}", flush=True)

    print("[1/3] 预处理（ASR + 音符转写）...", flush=True)
    t0 = time.time()
    res = preprocess_audio(
        args.audio,
        save_dir=workdir,
        language=args.language,
        vocal_sep=args.vocal_sep,
    )
    print(f"      耗时 {time.time()-t0:.1f}s", flush=True)

    metadata = load_metadata(res.metadata_path)
    if structure is None:
        structure = note_structure(metadata)
        title = "原词结构（每行一句，~ 表示转音）"
    else:
        try:
            metadata = apply_structure(metadata, structure, language=args.language)
        except ValueError as e:
            print("---- 自动识别的原词结构 ----")
            print(structure_text(note_structure(metadata)))
            sys.exit(f"原词结构与音频不符：{e}")
        save_metadata(metadata, os.path.join(workdir, "corrected_metadata.json"))
        title = "原词结构（已应用校正）"
    total_clean, per_phrase = count_clean_notes(metadata, structure=structure)
    print(f"---- {title}：{total_clean} 字，共 {len(per_phrase)} 句 ----")
    print(structure_text(structure))
    if new_lyrics is None:
        print("未指定 --new-lyrics，仅识别。", flush=True)
        return

    print("[2/3] 歌词替换...", flush=True)
    try:
        new_metadata = replace_lyrics(
            metadata, new_lyrics, language=args.language, structure=structure
        )
    except ValueError as e:
        sys.exit(f"新歌词与原词结构不符：{e}")
    new_meta_path = os.path.join(workdir, "edit_metadata.json")
    save_metadata(new_metadata, new_meta_path)

    prompt_wav, prompt_metadata = res.vocal_path, metadata
    ref = None
    if args.ref_audio:
        print("      预处理参考音色...", flush=True)
        ref = preprocess_reference(os.path.abspath(args.ref_audio), os.path.join(workdir, "reference"), args.language)
        prompt_wav, prompt_metadata = ref.vocal_path, load_metadata(ref.metadata_path)
        if not prompt_metadata:
            sys.exit("参考音频未识别出歌词，无法用于改词合成（--svc 只换音色仍可用）")
    shift = _pitch_shift(args, ref, res)

    print("[3/3] SoulX-Singer 合成...", flush=True)
    engine = SoulXSingerEngine(
        model_path=os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model.pt"),
        config_path=os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml"),
        phoneset_path=os.path.join(SOULX, "soulxsinger", "utils", "phoneme", "phone_set.json"),
    )
    t0 = time.time()
    audio, sr = engine.synthesize(
        prompt_wav_path=prompt_wav,
        prompt_metadata=prompt_metadata,
        target_metadata=new_metadata,
        control=args.control,
        auto_shift=False,  # 上游逐段计算，改为全曲统一的 shift
        pitch_shift=shift,
    )
    print(f"      合成耗时 {time.time()-t0:.1f}s", flush=True)

    sf.write(args.output, audio, sr)
    print(f"完成 -> {args.output}", flush=True)


def _pitch_shift(args, ref, src):
    """总移调 = 自动（有参考音色且未关闭时，全曲统一）+ 手动。"""
    auto = 0
    if ref is not None:
        auto = estimate_pitch_shift(np.load(ref.f0_path), np.load(src.f0_path), args.auto_shift)
    total = auto + args.pitch_shift
    parts = [f"自动[{args.auto_shift}] {auto:+d}"] if ref is not None and args.auto_shift != "off" else []
    if args.pitch_shift:
        parts.append(f"手动 {args.pitch_shift:+d}")
    print(f"      变调 {total:+d} 个半音（{'，'.join(parts)}）" if parts else "      未变调", flush=True)
    return total


def run_svc(args):
    """只换音色：原曲歌词、旋律、节奏不变，音色换成参考音色（不做歌词识别）。"""
    workdir = new_run_dir(os.path.join(BASE, "outputs"), args.audio)
    print(f"工作目录：{workdir}", flush=True)
    print("[1/2] 预处理（提取人声 F0）...", flush=True)
    src = preprocess_audio(
        args.audio, save_dir=workdir, language=args.language, vocal_sep=args.vocal_sep, midi_transcribe=False
    )
    ref = preprocess_reference(
        os.path.abspath(args.ref_audio), os.path.join(workdir, "reference"), args.language, transcribe=False
    )
    shift = _pitch_shift(args, ref, src)
    print("[2/2] SoulX-Singer SVC 音色转换...", flush=True)
    t0 = time.time()
    audio, sr = SoulXSingerSVCEngine().convert(
        prompt_wav_path=ref.vocal_path,
        prompt_f0_path=ref.f0_path,
        target_wav_path=src.vocal_path,
        target_f0_path=src.f0_path,
        pitch_shift=shift,
    )
    print(f"      转换耗时 {time.time()-t0:.1f}s", flush=True)
    sf.write(args.output, audio, sr)
    print(f"完成 -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
