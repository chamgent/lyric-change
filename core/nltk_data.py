# -*- coding: utf-8 -*-
"""英文注音（g2p_en）所需的 NLTK 数据。

g2p_en 在 import 时只会自动下载旧名 averaged_perceptron_tagger，而 nltk>=3.9 的
pos_tag 需要 averaged_perceptron_tagger_eng；缺失时任何英文单词注音都会抛 LookupError。
"""
import nltk

_RESOURCES = {
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/cmudict": "cmudict",
}


def ensure_english_g2p_data(download=True):
    """检查（并按需下载）英文注音所需的 NLTK 数据，返回是否齐全。"""
    for path, package in _RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            if not (download and nltk.download(package, quiet=True)):
                return False
    return True
