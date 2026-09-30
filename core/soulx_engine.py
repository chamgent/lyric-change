# -*- coding: utf-8 -*-
"""SoulX-Singer 引擎封装：加载模型 + 受控歌声合成。

复用官方 cli/inference.py 的模型构建与推理逻辑，封装为可长期驻留的
单例引擎，供 Gradio WebUI 与 CLI 复用。
"""
import copy
import json
import os
import sys

import numpy as np
import torch

SOULX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SoulX-Singer"
)
if SOULX_ROOT not in sys.path:
    sys.path.insert(0, SOULX_ROOT)

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.utils.data_processor import DataProcessor


class SoulXSingerEngine:
    def __init__(
        self,
        model_path,
        config_path,
        phoneset_path,
        device="cuda",
        use_fp16=True,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.phoneset_path = phoneset_path
        self.device = device
        self.use_fp16 = use_fp16

        self.config = load_config(config_path)
        self.model = self._build_model()
        self.data_processor = DataProcessor(
            hop_size=self.config.audio.hop_size,
            sample_rate=self.config.audio.sample_rate,
            phoneset_path=self.phoneset_path,
            device=self.device,
        )

    def _build_model(self):
        model = SoulXSinger(self.config).to(self.device)
        checkpoint = torch.load(self.model_path, weights_only=False, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError("checkpoint 缺少 state_dict")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if self.use_fp16 and self.device.startswith("cuda"):
            model.half()
            model.mel.float()
        model.eval()
        model.to(self.device)
        return model

    @torch.no_grad()
    def synthesize(
        self,
        prompt_wav_path,
        prompt_metadata,
        target_metadata,
        control="score",
        auto_shift=False,
        pitch_shift=0,
    ):
        """根据 prompt(音色) + target(新词元数据) 合成歌声。

        prompt_metadata / target_metadata: 可为 metadata.json 路径或 list[dict]
        返回: (audio_np: float32 mono, sample_rate: int)
        """
        prompt_metadata = self._ensure_loaded(prompt_metadata)
        target_metadata = self._ensure_loaded(target_metadata)

        prompt_meta = prompt_metadata[0]
        infer_prompt_data = self.data_processor.process(prompt_meta, prompt_wav_path)

        sample_rate = self.config.audio.sample_rate
        generated_len = int(target_metadata[-1]["time"][1] / 1000 * sample_rate)
        generated = np.zeros(generated_len, dtype=np.float32)

        for target_meta in target_metadata:
            start_idx = int(target_meta["time"][0] / 1000 * sample_rate)
            infer_target_data = self.data_processor.process(target_meta, None)
            infer_data = {"prompt": infer_prompt_data, "target": infer_target_data}

            audio = self.model.infer(
                infer_data,
                auto_shift=auto_shift,
                pitch_shift=pitch_shift,
                n_steps=self.config.infer.n_steps,
                cfg=self.config.infer.cfg,
                control=control,
                use_fp16=self.use_fp16,
            )
            audio = audio.squeeze().cpu().numpy()
            n = min(audio.shape[0], generated.shape[0] - start_idx)
            generated[start_idx : start_idx + n] = audio[:n]

        return generated, sample_rate

    @staticmethod
    def _ensure_loaded(meta):
        # 上游 DataProcessor.merge_phoneme 会原地把 duration/phoneme 等字段改写成 list，
        # 必须深拷贝，否则调用方缓存的 metadata（如 WebUI 的 _state）在首次合成后即被破坏。
        if isinstance(meta, str):
            with open(meta, encoding="utf-8") as f:
                return json.load(f)
        return copy.deepcopy(meta)
