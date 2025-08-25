
# Flux + LoRA 模型部署指南

本指南详细介绍如何使用 **Ssy** 将基于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 和自定义节点的 Stable Diffusion 工作流打包为 Docker 容器，并部署至 [胜算云](https://www.shengsuanyun/) 平台。

本指南适用于有一定 Python、Docker 基础，并希望快速在云端部署 Flux + LoRA 模型的用户。

---

## 环境要求

在开始之前，请确保满足以下条件：

| 依赖     | 说明                                                                |
| ------ | ----------------------------------------------------------------- |
| 系统     | Linux / macOS（Windows 用户可使用 WSL2）                                 |
| Docker | Ssy 依赖 Docker 运行容器。安装并启动 Docker：<br>`bash docker info`            |
| 模型权重   | 包括 LoRA 模型、基础模型等，放在 ComfyUI 的 `models` 目录下                        |
| 胜算云账号  | 登录 [胜算云](https://console.shengsuanyun.com/user/keys)，获取 API Token |

---

## 部署流程概览

部署分为两个阶段：

1. **本地环境搭建与测试**

   * 配置 ComfyUI
   * 安装自定义节点
   * 测试工作流
2. **使用 Ssy 打包与部署**

   * 将本地测试好的环境打包为 Docker 容器
   * 推送到胜算云并测试 API

> 直接使用示例可参考：[https://github.com/xinrui-z/ComfyUI_Flux_LoRA](https://github.com/xinrui-z/ComfyUI_Flux_LoRA)

---

## 第一部分：本地 ComfyUI 环境搭建

### 1. 安装 ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows 用户可使用：
# venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 安装自定义节点

安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) 和 [nunchaku 节点](https://github.com/mit-han-lab/ComfyUI-nunchaku)：

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

> ⚠️ 注意：根据 Python 版本和 CUDA 环境选择合适的 `.whl` 文件。

### 4. 下载模型权重

将模型文件放入 `ComfyUI/models` 下对应目录：

| 模型         | 来源                                                                                      | 放置路径示例                                                 |
| ---------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| FLUX.1-vae | [Hugging Face](https://huggingface.co/diffusers/FLUX.1-vae/tree/main)                   | `ComfyUI/models/vae/FLUX.1-vae.safetensors`            |
| FLUX-FP8   | [Hugging Face](https://huggingface.co/Kijai/flux-fp8/tree/main)                         | `ComfyUI/models/diffusion_models/FLUX-FP8.safetensors` |
| Flymy LoRA | [Hugging Face](https://huggingface.co/flymy-ai/qwen-image-realism-lora/tree/main)       | `ComfyUI/models/loras/Flymy.safetensors`               |
| t5xxl\_fp8 | [Hugging Face](https://huggingface.co/fmoraes2k/t5xxl_fp8_e4m3fn.safetensors/tree/main) | `ComfyUI/models/text_encoders/t5xxl_fp8.safetensors`   |

---

### 5. 启动并测试工作流

```bash
cd .. # 返回 ComfyUI 根目录
python main.py --listen 0.0.0.0
```

* 访问 `http://<你的机器IP>:8188`
* 加载工作流 JSON 文件（例如 `flux_lora_api.json`）
* 确认所有节点和模型正确加载，并能生成图片
* 记录 API 参数节点（如 `seed`, `prompt`, `lora_strength`），后续在 `predict.py` 中使用

> 示例工作流：[flux\_lora\_api.json](https://github.com/xinrui-z/Cog_ComfyUI_SD_LoRA/blob/main/examples/api_workflows/fluc_lora_api.json)

---

## 第二部分：使用 Ssy 打包部署

### 1. 安装 Ssy (仅支持 x86)

```bash
sudo curl -o /usr/local/bin/ssy -L "https://shengsuanyun.oss-cn-shanghai.aliyuncs.com/ssy/ssy"
sudo chmod +x /usr/local/bin/ssy
```

### 2. 初始化 Ssy 项目

```bash
git clone --recurse-submodules https://github.com/replicate/cog-comfyui.git my-comfyui-app
cd my-comfyui-app
```

### 3. 替换并配置 ComfyUI

```bash
rm -rf ComfyUI
cp -r /path/to/your/fully-tested/ComfyUI ./
```

### 4. 配置 `ssy.yaml` 与 `predict.py`

**ssy.yaml 示例**：

```yaml
build:
  gpu: true
  python_version: "3.10"
  python_requirements: requirements.txt
  system_packages:
    - curl
  run:
    - "curl -o /usr/local/bin/pget -L \"https://gh-proxy.com/https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)\""
    - "chmod +x /usr/local/bin/pget"
    - "mkdir -p /root/.cache/torch/hub/checkpoints"
    - "pget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth /root/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"

predict: "predict.py:Predictor"
```

> ⚠️ 注意：请确保已登录胜算云账号并获取 API Token。

**predict.py 核心功能**：

* 自动下载并管理用户权重
* 加载 ComfyUI 工作流并修改关键参数
* 批量生成图片并优化输出质量

> ⚠️ 节点 ID 必须与 workflow JSON 中一致，否则可能运行失败。

### 5. 本地测试 Ssy 构建

```bash
ssy build
ssy predict -i prompt="a beautiful landscape"
```

### 6. 部署到胜算云

1. 在胜算云 [模型超市](https://www.shengsuanyun.com/model) 创建模型
2. 推送模型到云端：

```bash
ssy login
ssy push your-username/your-model-name
```

成功推送后，可在胜算云页面测试 API 并获取调用代码。


