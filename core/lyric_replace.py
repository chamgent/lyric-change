# -*- coding: utf-8 -*-
"""歌词替换模块：把新歌词注入原 metadata，保持每字时值/音高不变。

核心原理：metadata 中每个音符 = 一个音素 = 一个汉字。改词时只替换
非 <SP> 的 text/phoneme，duration / note_pitch / note_type / f0 原样保留，
从而做到“每字时值严格等于原曲节奏”。

对齐粒度：按“句”对齐。metadata 里的 <SP> 是停顿标记，只有「长停顿」
（>= sp_threshold）才算句边界；短 <SP> 是一字多音中间的换气，不切句。
这样显示和填词共用同一套句结构，天然一致。

note_type 语义（来自 ROSVOT note2words 对齐）：
  1 = <SP>（停顿/换气）
  2 = 词首音符（onset）
  3 = 同词续音（转音/一字多音，续音不算新字）
"""
import copy
import json
import os
import re
import sys

SOULX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer"
)
if SOULX_ROOT not in sys.path:
    sys.path.insert(0, SOULX_ROOT)

from preprocess.tools.g2p import g2p_transform

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

SP_THRESHOLD = 0.25


def extract_chars(text):
    """从用户输入中按序抽取所有汉字（忽略标点/空白/英文）。"""
    return _CJK_RE.findall(text)


def _note_phrases(metadata_list, sp_threshold=SP_THRESHOLD):
    """把 metadata 按「长停顿」切成句。

    返回 list[list[dict]]，每句是一串音符，{"tok": 字, "onset": 是否词首}。
    onset = note_type != 3（转音续音不算词首）。短 <SP>（<阈值）不切句。
    """
    phrases = []
    for seg in metadata_list:
        tokens = seg["text"].split()
        types = seg["note_type"].split()
        durs = seg["duration"].split()
        cur = []
        for tok, t, d in zip(tokens, types, durs):
            if tok == "<SP>":
                if cur and float(d) >= sp_threshold:
                    phrases.append(cur)
                    cur = []
            else:
                cur.append({"tok": tok, "onset": (t != "3")})
        if cur:
            phrases.append(cur)
    return phrases


def count_notes(metadata_list):
    """统计 metadata 中所有非 <SP> 的音符数（含转音重复字，逐音符计数）。"""
    total = 0
    per_seg = []
    for seg in metadata_list:
        n = sum(1 for t in seg["text"].split() if t != "<SP>")
        per_seg.append(n)
        total += n
    return total, per_seg


def count_clean_notes(metadata_list, sp_threshold=SP_THRESHOLD):
    """返回 (去重后总字数, 每句字数列表)。用户按句填写，每句字数 = 该句词首音符数。"""
    counts = [
        sum(1 for n in ph if n["onset"]) for ph in _note_phrases(metadata_list, sp_threshold)
    ]
    return sum(counts), counts


def original_text(metadata_list):
    """返回原词全文（含转音重复字，按段用换行分隔，<SP> 显示为空格）。"""
    lines = []
    for seg in metadata_list:
        tokens = seg["text"].split()
        lines.append("".join(" " if t == "<SP>" else t for t in tokens))
    return "\n".join(lines)


def clean_text(metadata_list, sp_threshold=SP_THRESHOLD):
    """返回去重后的原词，每句一行（长停顿处换行），用于展示。"""
    return "\n".join(
        "".join(n["tok"] for n in ph if n["onset"])
        for ph in _note_phrases(metadata_list, sp_threshold)
    )


def replace_lyrics(metadata_list, new_lyrics_text, language="Mandarin", sp_threshold=SP_THRESHOLD):
    """按句把干净歌词注入 metadata（转音自动重复），返回新的 metadata list。

    new_lyrics_text: 新歌词，每行对应一句（行数与原词句数一致，空行忽略）。
    每行汉字数 == 该句词首音符数；每句内的转音（续音）自动重复对应汉字。
    某句字数不对会直接报出“第 N 句”，不会串到后面的句子。
    """
    lines = [l.strip() for l in (new_lyrics_text or "").splitlines() if l.strip()]
    phrases = _note_phrases(metadata_list, sp_threshold)

    if len(lines) != len(phrases):
        raise ValueError(f"句数不匹配：原词 {len(phrases)} 句，你写了 {len(lines)} 行")

    phrase_chars = []
    for i, (line, ph) in enumerate(zip(lines, phrases)):
        chars = extract_chars(line)
        need = sum(1 for n in ph if n["onset"])
        if len(chars) != need:
            raise ValueError(f"第 {i + 1} 句字数不匹配：原 {need} 字，你写 {len(chars)} 字")
        phrase_chars.append(chars)

    result = []
    pi = -1
    for seg in metadata_list:
        seg = copy.deepcopy(seg)
        tokens = seg["text"].split()
        types = seg["note_type"].split()
        durs = seg["duration"].split()
        new_tokens = []
        cur = ""
        in_phrase = False
        ci = 0
        for tok, t, d in zip(tokens, types, durs):
            if tok == "<SP>":
                new_tokens.append("<SP>")
                if in_phrase and float(d) >= sp_threshold:
                    in_phrase = False
            else:
                if t != "3":
                    if not in_phrase:
                        in_phrase = True
                        pi += 1
                        ci = 0
                    cur = phrase_chars[pi][ci]
                    ci += 1
                new_tokens.append(cur)
        seg["text"] = " ".join(new_tokens)
        seg["phoneme"] = " ".join(g2p_transform(new_tokens, language))
        result.append(seg)

    return result


def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata_list, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
