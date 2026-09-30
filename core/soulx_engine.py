# -*- coding: utf-8 -*-
"""SoulX-Singer 引擎封装：加载模型 + 受控歌声合成 / 歌声音色转换。

复用官方 cli/inference.py、cli/inference_svc.py 的模型构建与推理逻辑，封装为
可长期驻留的单例引擎，供 Gradio WebUI 与 CLI 复用：
  SoulXSingerEngine     改词合成（SVS）：prompt 音色 + 新词 metadata -> 歌声
  SoulXSingerSVCEngine  只换音色（SVC）：把一段演唱直接转成 prompt 的音色，不需要歌词
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
from soulxsinger.utils.audio_utils import load_wav
from soulxsinger.utils.data_processor import DataProcessor

MODELS_DIR = os.path.join(SOULX_ROOT, "pretrained_models", "SoulX-Singer")
SVC_MODEL_PATH = os.path.join(MODELS_DIR, "model-svc.pt")
CONFIG_PATH = os.path.join(SOULX_ROOT, "soulxsinger", "config", "soulxsinger.yaml")


AUTO_SHIFT_MODES = ("octave", "semitone", "off")


def estimate_pitch_shift(prompt_f0, target_f0, mode="octave"):
    """把目标旋律移到 prompt 歌手音域所需的半音数（两段有声 F0 中位数之比）。

    mode:
      "octave"   只按整八度移（保持原调，仍能配原曲伴奏；如女声转男声通常降 12）
      "semitone" 精确到半音，最贴合 prompt 音域，但歌曲整体会变调（与上游 auto_shift 相同）
      "off"      不自动变调
    与上游 auto_shift 不同，由调用方对整首歌只算一次：上游 SVS 的 auto_shift 在每个
    分段内单独计算，同一首歌的不同分段可能被移到不同的调。
    """
    if mode not in AUTO_SHIFT_MODES:
        raise ValueError(f"未知的自动变调模式: {mode}")
    if mode == "off":
        return 0
    pt = np.asarray(prompt_f0, dtype=np.float64)
    gt = np.asarray(target_f0, dtype=np.float64)
    pt, gt = pt[pt > 0], gt[gt > 0]
    if len(pt) == 0 or len(gt) == 0:
        return 0
    semitones = 12 * np.log2(np.median(pt) / np.median(gt))
    if mode == "octave":
        return int(12 * np.round(semitones / 12))
    return int(np.round(semitones))


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


class SoulXSingerSVCEngine:
    """只换音色（SVC）：保留原演唱的歌词、旋律与节奏，只把音色换成 prompt。"""

    def __init__(self, model_path=SVC_MODEL_PATH, config_path=CONFIG_PATH, device="cuda", use_fp16=True):
        from cli.inference_svc import build_model

        self.device = device
        self.use_fp16 = use_fp16 and device.startswith("cuda")
        self.config = load_config(config_path)
        self.model = build_model(model_path, self.config, device=device, use_fp16=self.use_fp16)

    @torch.no_grad()
    def convert(self, prompt_wav_path, prompt_f0_path, target_wav_path, target_f0_path, pitch_shift=0):
        """返回 (audio_np: float32 mono, sample_rate)。F0 为预处理产出的 *_f0.npy。"""
        sr = self.config.audio.sample_rate
        pt_wav = load_wav(prompt_wav_path, sr).to(self.device)
        gt_wav = load_wav(target_wav_path, sr).to(self.device)
        pt_f0 = torch.from_numpy(np.load(prompt_f0_path)).float().unsqueeze(0).to(self.device)
        gt_f0 = torch.from_numpy(np.load(target_f0_path)).float().unsqueeze(0).to(self.device)
        audio, _ = self.model.infer(
            pt_wav=pt_wav,
            gt_wav=gt_wav,
            pt_f0=pt_f0,
            gt_f0=gt_f0,
            auto_shift=False,
            pitch_shift=int(pitch_shift),
            n_steps=self.config.infer.n_steps,
            cfg=self.config.infer.cfg,
            use_fp16=self.use_fp16,
        )
        return audio.squeeze().float().cpu().numpy(), sr
