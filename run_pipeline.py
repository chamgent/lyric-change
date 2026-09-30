# -*- coding: utf-8 -*-
"""端到端 CLI：干声 + 新歌词 -> 新干声。

用法:
    python run_pipeline.py --audio 干声.wav --new-lyrics 新词.txt --output out.wav

新词文件：每行一句（与原词句数/每句字数一致，转音自动补全）。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.soulx_preprocess import new_run_dir, preprocess_audio
from core.lyric_replace import (
    load_metadata,
    replace_lyrics,
    count_clean_notes,
    clean_text,
    save_metadata,
)
from core.soulx_engine import SoulXSingerEngine

BASE = os.path.dirname(os.path.abspath(__file__))
SOULX = os.path.join(BASE, "SoulX-Singer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="原曲干声")
    ap.add_argument("--new-lyrics", required=True, help="新歌词文件(UTF-8)")
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
    args = ap.parse_args()

    args.audio = os.path.abspath(args.audio)
    args.new_lyrics = os.path.abspath(args.new_lyrics)
    args.output = os.path.abspath(args.output)

    with open(args.new_lyrics, encoding="utf-8") as f:
        new_lyrics = f.read()

    workdir = new_run_dir(os.path.join(BASE, "outputs"), args.audio)
    print(f"工作目录：{workdir}", flush=True)

    print("[1/3] 预处理（ASR + 音符转写）...", flush=True)
    t0 = time.time()
    metadata_path, vocal_path = preprocess_audio(
        args.audio,
        save_dir=workdir,
        language=args.language,
        vocal_sep=args.vocal_sep,
    )
    print(f"      耗时 {time.time()-t0:.1f}s", flush=True)

    metadata = load_metadata(metadata_path)
    total_clean, per_phrase = count_clean_notes(metadata)
    print(f"[2/3] 歌词替换（去重后原词 {total_clean} 字，共 {len(per_phrase)} 句）...", flush=True)
    print("---- 原词（每行一句） ----")
    print(clean_text(metadata))

    new_metadata = replace_lyrics(metadata, new_lyrics, language=args.language)
    new_meta_path = os.path.join(workdir, "edit_metadata.json")
    save_metadata(new_metadata, new_meta_path)

    print("[3/3] SoulX-Singer 合成...", flush=True)
    engine = SoulXSingerEngine(
        model_path=os.path.join(SOULX, "pretrained_models", "SoulX-Singer", "model.pt"),
        config_path=os.path.join(SOULX, "soulxsinger", "config", "soulxsinger.yaml"),
        phoneset_path=os.path.join(SOULX, "soulxsinger", "utils", "phoneme", "phone_set.json"),
    )
    t0 = time.time()
    audio, sr = engine.synthesize(
        prompt_wav_path=vocal_path,
        prompt_metadata=metadata,
        target_metadata=new_metadata,
        control=args.control,
        auto_shift=False,
        pitch_shift=0,
    )
    print(f"      合成耗时 {time.time()-t0:.1f}s", flush=True)

    import soundfile as sf

    sf.write(args.output, audio, sr)
    print(f"完成 -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
