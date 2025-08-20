# Cog-ComfyUI 部署 Stable Diffusion + LoRA 模型指南

本指南详细介绍如何使用 [Cog](https://github.com/replicate/cog) 将基于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 和自定义节点的 Stable Diffusion 工作流打包为 Docker 容器，并部署至 [Replicate](https://replicate.com/) 平台。

---

## Requirements

在开始之前，请确保满足以下条件：

| 依赖           | 说明                                                         |
| ------------ | ---------------------------------------------------------- |
| 系统           | Linux/macOS (Windows 可使用 WSL2)                             |
| Docker       | Cog 依赖 Docker 运行容器。安装并启动 Docker。<br>`bash docker info `    |
| 模型权重         | 包括 LoRA 模型、基础模型等，放在 ComfyUI 的 `models` 目录下                 |
| Replicate 账号 | [注册 Replicate](https://replicate.com/signin) 并获取 API Token |

---

## 部署流程概览

部署分为两个阶段：

1. **本地环境搭建与测试**
   配置 ComfyUI、安装自定义节点、测试工作流。
2. **使用 Cog 打包与部署**
   将本地测试好的环境打包为 Docker 容器并推送到 Replicate。

---

## 第一部分：本地 ComfyUI 环境搭建

### 1. 安装 ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 安装自定义节点

安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) 和 [nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku) 节点：

```bash
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git
```

### 3. 安装 Nunchaku 后端库

```bash
# 在 ComfyUI 虚拟环境中执行
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl
```

> 注意：请根据 Python 版本和 CUDA 环境，选择合适的 `.whl` 文件。

### 4. 下载模型权重

将模型文件放入 `ComfyUI/models` 下对应目录：

| 模型         | 来源                                                                                      | 路径                           |
| ---------- | --------------------------------------------------------------------------------------- | ---------------------------- |
| FLUX.1-vae | [Hugging Face](https://huggingface.co/diffusers/FLUX.1-vae/tree/main)                   | `./models/vae/`              |
| FLUX-FP8   | [Hugging Face](https://huggingface.co/Kijai/flux-fp8/tree/main)                         | `./models/diffusion_models/` |
| Flymy LoRA | [Hugging Face](https://huggingface.co/flymy-ai/qwen-image-realism-lora/tree/main)       | `./models/loras/`            |
| t5xxl\_fp8 | [Hugging Face](https://huggingface.co/fmoraes2k/t5xxl_fp8_e4m3fn.safetensors/tree/main) | `./models/text_encoders/`    |

### 5. 启动并测试工作流

```bash
cd .. # 返回 ComfyUI 根目录
python main.py --listen 0.0.0.0
```

* 访问 `http://<你的机器IP>:8188`
* 加载工作流 JSON 文件（例如 `sd_lora_api.json`）
* 确认所有节点和模型正确加载，并能生成图片
* 记录 API 参数节点（如 `seed`, `prompt`, `lora_strength`），在 `predict.py` 中使用

> 示例工作流：[sd\_lora\_api.json](./examples/api_workflows/sd_lora_api.json)

---

## 第二部分：使用 Cog 打包部署

### 1. 安装 Cog

```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

### 2. 初始化 Cog 项目

```bash
git clone --recurse-submodules https://github.com/replicate/cog-comfyui.git my-comfyui-app
cd my-comfyui-app
```

### 3. 替换并配置 ComfyUI

```bash
rm -rf ComfyUI
cp -r /path/to/your/fully-tested/ComfyUI ./
```

### 4. 配置 `cog.yaml` 与 `predict.py`

**cog.yaml 示例**：

```yaml
build:
  gpu: true
  cuda: "11.8"
  cudnn: "8"
  python_version: "3.10"
  python_packages:
    - "torch==2.0.1+cu118"
    - "torchvision==0.15.2+cu118"
    - "nunchaku==0.3.1+torch2.7"
    - "https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp310-cp310-linux_x86_64.whl"
  system_packages:
    - "libgl1"
    - "libglib2.0-0"
  run:
    - "cd ComfyUI && pip install -r requirements.txt"
    - "./scripts/install_custom_nodes.py"

predict: "predict.py:Predictor"
```

**predict.py 示例**：

```python
import os
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server("--listen 127.0.0.1 --port 8188")
        self.workflow = self.comfyUI.load_workflow("workflow_api.json")

    def predict(
        self,
        prompt: str = Input(description="正向提示"),
        negative_prompt: str = Input(description="负向提示", default=""),
        seed: int = Input(description="随机种子", default=-1),
        lora_strength: float = Input(description="LoRA 权重", default=0.8, ge=0.0, le=2.0),
    ) -> Path:
        self.workflow["6"]["inputs"]["text"] = prompt
        self.workflow["7"]["inputs"]["text"] = negative_prompt
        self.workflow["3"]["inputs"]["seed"] = seed
        self.workflow["10"]["inputs"]["strength_model"] = lora_strength

        output_path = self.comfyUI.run_workflow(self.workflow)
        return Path(output_path)
```

> 注意：节点 ID 必须与 `workflow_api.json` 中一致，可通过 UI 或脚本确认。

### 5. 本地测试 Cog 构建

```bash
cog build
cog predict -i prompt="a beautiful landscape" -i seed=42
```

### 6. 部署到 Replicate

```bash
cog login
cog push r8.im/your-username/your-model-name
```

成功推送后，可在 Replicate 页面测试 API 并获取调用代码。
