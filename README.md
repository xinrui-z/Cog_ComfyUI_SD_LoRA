## 环境要求

在开始之前，请确保满足以下条件：

| 依赖     | 说明                                                                |
| ------ | ----------------------------------------------------------------- |
| 系统     | Linux / macOS（Windows 可使用 WSL2）                                   |
| Docker | 需要 Docker 运行容器。安装后可用 `docker info` 验证                             |
| 模型权重   | 包括 LoRA 模型和基础模型等，请放在 ComfyUI 的 `models` 目录下                       |
| 胜算云账号  | 登录 [胜算云](https://console.shengsuanyun.com/user/keys) 获取 API Token |

---

## 克隆仓库

```bash
git clone https://github.com/xinrui-z/ComfyUI_Flux_LoRA.git
cd ComfyUI_Flux_LoRA
```

---

## 下载模型权重

将模型文件放入 `ComfyUI/models` 对应目录：

| 模型         | 来源                                                                                      | 路径                           |
| ---------- | --------------------------------------------------------------------------------------- | ---------------------------- |
| FLUX.1-vae | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/vae/diffusion_pytorch_model.safetensors)                   | `./models/vae/`              |
| FLUX.1-FP8   | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors)                         | `./models/diffusion_models/` |
| Flux-LoRA | [Hugging Face](https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main)       | `./models/loras/`            |
| t5xxl\_fp8 | [Hugging Face](https://huggingface.co/fmoraes2k/t5xxl_fp8_e4m3fn.safetensors/tree/main) | `./models/text_encoders/`    |
| FLUX.1-Text-Encoder | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/text_encoder/model.safetensors) | `./models/text_encoders/`    |

下载完成后可执行预测：

```bash
ssy predict
```

---

## 使用胜算云推送模型

1. 在 [胜算云模型超市](https://www.shengsuanyun.com/model) 创建模型。
2. 登录胜算云：

```bash
ssy login
```

> API Token 可在 [控制台](https://console.shengsuanyun.com/user/keys) 获取。

3. 推送模型：

```bash
ssy push your-username/your-model-name
```

推送成功后，可在胜算云页面测试 API 并获取调用代码。


