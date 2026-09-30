# -*- coding: utf-8 -*-
"""歌词替换模块：把新歌词注入原 metadata，保持每字时值/音高不变。

核心原理：metadata 中每个音符 = 一个音素 = 一个汉字。改词时只替换
非 <SP> 的 text/phoneme，duration / note_pitch / f0 原样保留，
从而做到“每字时值严格等于原曲节奏”。

对齐粒度：按“句”对齐。metadata 里的 <SP> 是停顿标记，只有「长停顿」
（>= sp_threshold）才算句边界；短 <SP> 是一字多音中间的换气，不切句。
这样显示和填词共用同一套句结构，天然一致。

新歌词可夹英文单词：上游格式里一个英文单词占一个音符，但中文一个音符只唱一个
音节，多音节单词直接塞进一个音符会被挤成一团（实测 forever 被唱成 fed）。所以
英文单词按音节数占用词首音符：第一个音符写该词，其余音节的音符标为续音
（note_type 3，与上游英文数据中一词跨多个音符的表示相同）。

原词结构（structure）：list[list[str]]，每句一个列表，每个非 <SP> 音符一个符号：
词首音符为该字，续音为 CONT（"~"）。自动识别可能分错句、错判转音或识别错字，
用户可以编辑结构文本（见 structure_text / parse_structure）进行校正，
apply_structure 再把校正写回 metadata。

note_type 语义（来自 ROSVOT note2words 对齐）：
  1 = <SP>（停顿/换气）
  2 = 词首音符（onset）
  3 = 同词续音（转音/一字多音，续音不算新字）
"""
import copy
import functools
import json
import os
import re
import sys

SOULX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer"
)
if SOULX_ROOT not in sys.path:
    sys.path.insert(0, SOULX_ROOT)

from preprocess.tools.g2p import g2p_english, g2p_transform

from .nltk_data import ensure_english_g2p_data

# 汉字，或英文单词（与上游 g2p 的英文单词规则一致，可带撇号，如 don't）
_UNIT_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)*")
_EXTRA_SYLLABLE = "\x00"  # 内部标记：多音节英文单词的后续音节

SP_THRESHOLD = 0.25

CONT = "~"
_STRUCT_SYMBOL_RE = re.compile(r"[\u4e00-\u9fff]|[~\uff5e]")  # 中文输入法打出的是全角「～」
_UNSUPPORTED_RE = re.compile(r"[A-Za-z0-9]")


def _is_english(unit):
    return unit[0].isascii()


def _require_english_g2p():
    if not ensure_english_g2p_data():
        raise ValueError("英文注音所需的 NLTK 数据缺失且自动下载失败，请联网后运行 python download_models.py")


@functools.lru_cache(maxsize=4096)
def english_syllables(word):
    """英文单词的音节数（ARPAbet 中带重音数字的元音个数，至少 1）。"""
    _require_english_g2p()
    return max(1, sum(p[-1].isdigit() for p in g2p_english(word.lower())))


def extract_units(text):
    """从用户输入中按序抽取填词单位：汉字或英文单词（忽略标点/空白/数字）。"""
    return _UNIT_RE.findall((text or "").replace("\u2019", "'"))  # 中文输入法的右单引号 ’


def unit_count(units):
    """一串填词单位占用的「字数」（词首音符数）：汉字 1 个，英文单词按音节数。"""
    return sum(english_syllables(u) if _is_english(u) else 1 for u in units)


def _expand_units(units):
    """把多音节英文单词展开为 [词, 标记, 标记, ...]，与词首音符一一对应。"""
    out = []
    for u in units:
        out.append(u)
        if _is_english(u):
            out.extend([_EXTRA_SYLLABLE] * (english_syllables(u) - 1))
    return out


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


def note_structure(metadata_list, sp_threshold=SP_THRESHOLD):
    """由 metadata 自动推出原词结构（每句的符号列表，续音为 CONT）。

    保证结果本身是合法结构：整首歌第一个音符不是续音；没有只含续音的句子
    （长停顿后紧跟续音的情况并入上一句）。
    """
    structure = []
    for ph in _note_phrases(metadata_list, sp_threshold):
        line = [n["tok"] if n["onset"] else CONT for n in ph]
        if not structure and line[0] == CONT:
            line[0] = ph[0]["tok"]
        if structure and all(s == CONT for s in line):
            structure[-1].extend(line)
        else:
            structure.append(line)
    return structure


def structure_text(structure):
    """原词结构 -> 可编辑文本：每句一行，续音显示为 ~。"""
    return "\n".join("".join(line) for line in structure)


def parse_structure(text):
    """解析用户编辑后的结构文本（每行一句，汉字为新字，~ 或 ～ 为续音）。

    标点和空白忽略；英文/数字暂不支持，直接报错以免悄悄错位。
    """
    structure = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        n = len(structure) + 1
        bad = _UNSUPPORTED_RE.search(raw)
        if bad:
            raise ValueError(f"第 {n} 行含有不支持的字符「{bad.group()}」（目前只支持汉字和 ~）")
        line = [CONT if s in "~～" else s for s in _STRUCT_SYMBOL_RE.findall(raw)]
        if not line:
            continue
        if all(s == CONT for s in line):
            raise ValueError(f"第 {n} 行只有 ~，没有字：请把这些转音并入上一行")
        structure.append(line)
    if not structure:
        raise ValueError("原词结构为空")
    if structure[0][0] == CONT:
        raise ValueError("第 1 行不能以 ~ 开头：~ 表示延续前一个字，但它前面没有字")
    return structure


def _inject(metadata_list, structure, phrase_chars, language):
    """按结构把每句的字写入 metadata（续音自动重复前一个字），返回新的 metadata list。

    同时按结构重写 note_type（词首 2 / 续音 3）并重新生成 phoneme；
    duration / note_pitch / f0 不变。结构的音符总数必须与 metadata 一致。
    """
    flat = [(pi, sym) for pi, line in enumerate(structure) for sym in line]
    n_notes = sum(1 for seg in metadata_list for t in seg["text"].split() if t != "<SP>")
    if len(flat) != n_notes:
        raise ValueError(f"原词结构共 {len(flat)} 个音（字 + ~），与音频的 {n_notes} 个音不一致")

    used = [0] * len(structure)
    k = 0
    cur = ""  # 跨分段延续：分段开头的续音延续上一段最后一个字
    result = []
    for seg in metadata_list:
        seg = copy.deepcopy(seg)
        new_tokens, new_types = [], []
        for tok, t in zip(seg["text"].split(), seg["note_type"].split()):
            if tok == "<SP>":
                new_tokens.append(tok)
                new_types.append(t)
                continue
            pi, sym = flat[k]
            k += 1
            if sym == CONT:
                new_types.append("3")
            else:
                unit = phrase_chars[pi][used[pi]]
                used[pi] += 1
                if unit == _EXTRA_SYLLABLE:  # 英文单词的后续音节：延续该词
                    new_types.append("3")
                else:
                    cur = unit
                    new_types.append("2")
            new_tokens.append(cur)
        seg["text"] = " ".join(new_tokens)
        seg["note_type"] = " ".join(new_types)
        seg["phoneme"] = " ".join(g2p_transform(new_tokens, language))
        result.append(seg)
    return result


def apply_structure(metadata_list, structure, language="Mandarin", sp_threshold=SP_THRESHOLD):
    """把（用户校正后的）原词结构写回原 metadata：改字、改转音、改分句。

    返回校正后的 metadata；它既用于填词对齐，也作为合成时的音色 prompt 元数据，
    所以改正识别错字也会改善合成。
    """
    total = sum(len(line) for line in structure)
    ref = note_structure(metadata_list, sp_threshold)
    need = sum(len(line) for line in ref)
    if total != need:
        msg = f"音的总数不对：原曲共 {need} 个音（字 + ~），现在是 {total} 个。只能改字、调整 ~ 的位置和换行，总数不能变"
        if len(structure) == len(ref):
            diffs = [
                f"第{i + 1}行{len(a) - len(b):+d}"
                for i, (a, b) in enumerate(zip(structure, ref))
                if len(a) != len(b)
            ]
            msg += "（" + "、".join(diffs) + "）"
        raise ValueError(msg)
    phrase_chars = [[s for s in line if s != CONT] for line in structure]
    return _inject(metadata_list, structure, phrase_chars, language)


def count_notes(metadata_list):
    """统计 metadata 中所有非 <SP> 的音符数（含转音重复字，逐音符计数）。"""
    total = 0
    per_seg = []
    for seg in metadata_list:
        n = sum(1 for t in seg["text"].split() if t != "<SP>")
        per_seg.append(n)
        total += n
    return total, per_seg


def count_clean_notes(metadata_list, sp_threshold=SP_THRESHOLD, structure=None):
    """返回 (去重后总字数, 每句字数列表)。用户按句填写，每句字数 = 该句词首音符数。

    structure: 校正后的原词结构；为 None 时由 metadata 自动推出。
    """
    if structure is None:
        structure = note_structure(metadata_list, sp_threshold)
    counts = [sum(1 for s in line if s != CONT) for line in structure]
    return sum(counts), counts


def original_text(metadata_list):
    """返回原词全文（含转音重复字，按段用换行分隔，<SP> 显示为空格）。"""
    lines = []
    for seg in metadata_list:
        tokens = seg["text"].split()
        lines.append("".join(" " if t == "<SP>" else t for t in tokens))
    return "\n".join(lines)


def replace_lyrics(
    metadata_list, new_lyrics_text, language="Mandarin", sp_threshold=SP_THRESHOLD, structure=None
):
    """按句把干净歌词注入 metadata（转音自动重复），返回新的 metadata list。

    new_lyrics_text: 新歌词，每行对应一句（行数与原词句数一致，空行忽略）。
    每行字数 == 该句词首音符数（英文单词按音节计）；每句内的转音（续音）自动重复对应的字/词。
    某句字数不对会直接报出“第 N 句”，不会串到后面的句子。
    structure: 校正后的原词结构；为 None 时由 metadata 自动推出。
    """
    if structure is None:
        structure = note_structure(metadata_list, sp_threshold)
    lines = [l.strip() for l in (new_lyrics_text or "").splitlines() if l.strip()]

    if len(lines) != len(structure):
        raise ValueError(f"句数不匹配：原词 {len(structure)} 句，你写了 {len(lines)} 行")

    phrase_chars = []
    for i, (line, st) in enumerate(zip(lines, structure)):
        units = extract_units(line)
        need = sum(1 for s in st if s != CONT)
        got = unit_count(units)
        if got != need:
            words = [f"{u}={english_syllables(u)}" for u in units if _is_english(u)]
            hint = f"（英文按音节计：{'、'.join(words)}）" if words else ""
            raise ValueError(f"第 {i + 1} 句字数不匹配：原 {need} 字，你写 {got} 字{hint}")
        phrase_chars.append(_expand_units(units))

    return _inject(metadata_list, structure, phrase_chars, language)


def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_metadata(metadata_list, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
