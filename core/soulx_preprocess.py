# -*- coding: utf-8 -*-
"""预处理封装：调用 SoulX-Singer 的 preprocess.pipeline（子进程）。

输入干声 -> 输出 metadata.json（含每字音高/时值/音素）。
"""
import json
import os
import subprocess
import sys
import time
import uuid

import jieba
import librosa
import soundfile as sf

SOULX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer"
)


def new_run_dir(outputs_root, audio_path):
    """为一次处理创建独立工作目录：<outputs_root>/<音频名>-<时间>-<随机>。

    只按文件名建目录会让同名音频、多次运行、多个用户互相覆盖中间文件和结果。
    """
    name = os.path.splitext(os.path.basename(audio_path))[0]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    path = os.path.join(outputs_root, f"{name}-{run_id}")
    os.makedirs(path)
    return path


def load_asr_text(save_dir, pause_threshold=0.25):
    """读取预处理产出的 ASR 原文（asr_words.json），按自然节奏换行返回。

    ASR 给的是「逐字 + 逐字时间戳」，字与字之间的间隙（<SP>）不一定是真正的
    停顿（唱歌时一个字一个音，字间常有短暂间隙）。这里用两条规则重排：

    1. 用 jieba 分词判断词边界，避免把一个词拦腰截断（如「记/得」「心/里」）。
    2. 只有间隙时长 >= pause_threshold 且正好落在词边界处，才换行。

    返回 None 表示 asr_words.json 不存在（旧 metadata 没有）。
    """
    p = os.path.join(save_dir, "asr_words.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        segs = json.load(f)

    chars = []
    gap_before = []
    pending = 0.0
    for s in segs:
        for w, d in zip(s.get("words", []), s.get("word_durs", [])):
            if w == "<SP>":
                pending += d
            else:
                chars.append(w)
                gap_before.append(pending)
                pending = 0.0

    if not chars:
        return ""

    boundaries = set()
    idx = 0
    for wd in jieba.lcut("".join(chars), HMM=False):
        idx += len(wd)
        boundaries.add(idx)

    lines = []
    cur = []
    for i, c in enumerate(chars):
        if i > 0 and gap_before[i] >= pause_threshold and i in boundaries and cur:
            lines.append("".join(cur))
            cur = []
        cur.append(c)
    if cur:
        lines.append("".join(cur))
    return "\n".join(lines)


def preprocess_audio(
    audio_path,
    save_dir,
    language="Mandarin",
    vocal_sep=False,
    midi_transcribe=True,
    max_merge_duration=15000,
    device="cuda",
):
    """运行预处理，返回 (metadata.json 路径, 预处理后的人声 vocal.wav 路径)。

    vocal.wav 与 metadata 的时间轴一致（开启 vocal_sep 时是分离后的纯人声），
    合成时应作为音色 prompt 使用，而不是原始输入。

    max_merge_duration: 合并后的单段最长时长(ms)。SoulX 模型对长片段会退化
    （中后段出现电流/咬字不清/时断时续），默认限制在 15s 以内，拆成多段分别合成。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 上游 pipeline 结束时会把 metadata.json 复制到「输入路径把 .wav/.mp3/.flac 替换成
    # .json」的位置：输入若是 .m4a/.ogg/大写 .WAV 等，替换不生效，用户的原音频会被
    # 覆盖；即使是 .wav 也会在用户目录里留下一个 .json。所以先转成工作目录内的 wav。
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    input_wav = os.path.join(save_dir, "input.wav")
    sf.write(input_wav, y.T, sr, subtype="FLOAT")

    cmd = [
        sys.executable,
        "-m",
        "preprocess.pipeline",
        "--audio_path", input_wav,
        "--save_dir", save_dir,
        "--language", language,
        "--device", device,
        "--vocal_sep", str(vocal_sep).lower(),
        "--max_merge_duration", str(max_merge_duration),
        "--midi_transcribe", str(midi_transcribe).lower(),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = SOULX_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        cmd,
        cwd=SOULX_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    if proc.returncode != 0:
        raise RuntimeError(f"preprocess 失败 (exit {proc.returncode}):\n{out[-4000:]}")

    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"preprocess 未产出 metadata.json:\n{out[-4000:]}")

    return metadata_path, os.path.join(save_dir, "vocal.wav")
