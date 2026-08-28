# -*- coding: utf-8 -*-
"""预处理封装：调用 SoulX-Singer 的 preprocess.pipeline（子进程）。

输入干声 -> 输出 metadata.json（含每字音高/时值/音素）。
"""
import json
import os
import subprocess
import sys

import jieba

SOULX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer"
)


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
    """运行预处理，返回 metadata.json 路径。

    max_merge_duration: 合并后的单段最长时长(ms)。SoulX 模型对长片段会退化
    （中后段出现电流/咬字不清/时断时续），默认限制在 15s 以内，拆成多段分别合成。
    """
    os.makedirs(save_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "preprocess.pipeline",
        "--audio_path", audio_path,
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

    return metadata_path
