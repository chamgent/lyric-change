# LyricChange (改词翻唱)

基于 **SoulX-Singer** 引擎的「改词翻唱」开源实现。

用户仅需提供 **① 原曲干声 ② 改后歌词**，即可一键生成保持 **原音色、原旋律、原节奏** 的全新干声。

---

## ✨ 核心特性

- **🎯 节奏与音高严格对齐**：改词时只替换文字与注音（`text` / `phoneme`），原曲每字的 `duration` / `note_pitch` / `f0` 严格保留，确保每字时值与旋律严格等于原曲。
- **🎵 转音（一字多音）自动补全**：自动识别歌曲中的转音与连音，用户只需写干净的正常歌词，系统自动在转音位置补全对应汉字，无需手动对着重复字数数。
- **✂️ 自然断句与逐句独立对齐**：基于自然呼吸停顿切句，界面显示与填词按句对齐，逐句独立校验。某句字数偏差仅提示该句，绝不引起后文雪崩式错位。
- **🎛️ 双控制模式切换**：
  - `melody`（默认）：跟随原唱真实 F0 曲线，完美还原滑音、颤音与转音细节。
  - `score`：跟随量化音符音高，更干净稳定。
- **🖥️ 极简 Gradio WebUI & CLI**：提供可视化网页交互（实时字数/行数检测、🟢/🔴 匹配提示）与自动化命令行脚本。

---

## 🛠️ 环境要求

- **操作系统**：Linux / Windows
- **Python**：3.10+
- **GPU**：NVIDIA GPU，显存建议 >= 12GB（如 RTX 3080 / 3090 / 4090）
- **PyTorch**：2.0+ (CUDA 11.8 / 12.1)

---

## 🚀 快速开始

### 方式一：Docker 一键部署（推荐）

需本机已安装 [Docker](https://docs.docker.com/engine/install/) 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（用于 GPU 透传）。

```bash
# 1. 克隆项目
git clone https://github.com/chamgent/lyric-change.git
cd lyric-change

# 2. 使用 Docker Compose 一键构建并启动（首次运行会自动下载模型）
docker compose up -d

# 3. 查看运行日志
docker compose logs -f
```

容器启动后，在浏览器访问 `http://localhost:7860` 即可使用。

---

### 方式二：Conda 本地环境部署

#### 1. 克隆本项目与上游依赖

```bash
git clone https://github.com/chamgent/lyric-change.git
cd lyric-change

# 克隆上游 SoulX-Singer 依赖库
git clone https://github.com/Soul-AILab/SoulX-Singer.git
```

#### 2. 创建环境与安装依赖

```bash
# 系统依赖：编译 webrtcvad 等需要 C 编译器，音频解码需要 ffmpeg
# Ubuntu/Debian: sudo apt install -y build-essential ffmpeg

conda create -n soulxsinger python=3.10 -y
conda activate soulxsinger

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt -c constraints.txt
```

#### 3. 一键下载预训练模型

运行内置的下载脚本（默认先尝试 `hf-mirror.com`，失败自动回退 `huggingface.co`；也可用 `HF_ENDPOINT` 环境变量指定下载源）：

```bash
python download_models.py
```

下载内容包括：
- `SoulX-Singer` 歌声合成主模型 (`model.pt`)
- `SoulX-Singer-Preprocess` 预处理套件（Paraformer ASR 识别、RMVPE 音高提取、ROSVOT 音符转写）

---

## 💻 使用方法

### 方式一：Gradio WebUI 网页交互（推荐）

启动 Web 服务：

```bash
python app.py
# 或使用后台启动脚本（需先 conda activate，或用 PYTHON=/path/to/python 指定解释器）：
# bash start_server_soulx.sh
```

打开浏览器访问 `http://localhost:7860`：

1. **上传原曲干声**：系统自动调用 ASR 与音符转写，在左侧提取出按句排版的原歌词；
2. **输入改后歌词**：在右侧输入框填入新歌词（每行对应一句），下方会实时显示匹配状态（如 `🟢 匹配（29 句，共 172 字）`）；
3. **选择音高模式**：选择 `melody`（更还原）或 `score`（更干净）；
4. **点击生成翻唱**：完成后即可在线试听并下载生成的新干声音频。

---

### 方式二：CLI 命令行端到端运行

准备好干声文件与新歌词文本（新词文本每行一句）：

```bash
python run_pipeline.py \
  --audio path/to/vocal.wav \
  --new-lyrics path/to/lyrics.txt \
  --output output_cover.wav \
  --control melody
```

可选参数：
- `--control`: 音高控制模式，可选 `melody`（默认）或 `score`
- `--language`: 语言，目前默认 `Mandarin`（中文普通话）
- `--vocal-sep`: 若输入的音频包含伴奏，开启此选项自动执行人声分离

---

## 📂 项目结构

```text
lyric-change/
├── app.py                  # Gradio WebUI 主程序
├── run_pipeline.py         # 命令行 CLI 端到端处理脚本
├── download_models.py      # 模型一键下载脚本
├── start_server_soulx.sh   # Linux 后台守护启动脚本
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker Compose 编排与 GPU 透传配置
├── docker-entrypoint.sh    # Docker 容器自动初始化入口
├── requirements.txt        # Python 依赖清单
├── core/
│   ├── lyric_replace.py    # 核心：歌词注入、转音展开、逐句对齐算法
│   ├── soulx_engine.py     # 核心：SoulX-Singer SVS 引擎单例封装
│   └── soulx_preprocess.py # 核心：音频预处理、ASR识别与分片合并
├── SoulX-Singer/           # 上游模型推理库
└── README.md
```

---

## 🤝 鸣谢与致谢

- [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)：提供强大的音符与 F0 受控歌声合成基座
- [FunASR / Paraformer](https://github.com/alibaba-damo-academy/FunASR)：高准确率中文歌词识别
- [RMVPE](https://github.com/Dream-High/RMVPE)：高精度鲁棒基频提取
- [ROSVOT](https://github.com/yxlu-0102/ROSVOT)：歌声转写与词音符对齐

---

## 📄 开源许可证

本项目核心代码遵循 [Apache 2.0 License](LICENSE)。使用 SoulX-Singer 及其预训练权重时请遵循其官方开源协议与非商业限制。
