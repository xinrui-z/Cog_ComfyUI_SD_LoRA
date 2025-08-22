import os
import shutil
import mimetypes
from typing import List, Optional
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from weights_downloader import WeightsDownloader
from cog_model_helpers import optimise_images
from config import config
import requests
import json

# 环境变量配置
os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# MIME 类型注册
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/webm", ".webm")

# 全局目录
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"  
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# 默认workflow（API 格式）
with open("examples/api_workflows/sd_lora_api.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self, weights: str = ""):
        if bool(weights):
            self.handle_user_weights(weights)

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def handle_user_weights(self, weights: str):
        if hasattr(weights, "url"):
            if weights.url.startswith("http"):
                weights_url = weights.url
            else:
                weights_url = "https://replicate.delivery/" + weights.url
        else:
            weights_url = weights

        print(f"Downloading user weights from: {weights_url}")
        WeightsDownloader.download("weights.tar", weights_url, config["USER_WEIGHTS_PATH"])
        for item in os.listdir(config["USER_WEIGHTS_PATH"]):
            source = os.path.join(config["USER_WEIGHTS_PATH"], item)
            destination = os.path.join(config["MODELS_PATH"], item)
            if os.path.isdir(source):
                if not os.path.exists(destination):
                    print(f"Moving {source} to {destination}")
                    shutil.move(source, destination)
                else:
                    for root, _, files in os.walk(source):
                        for file in files:
                            if not os.path.exists(os.path.join(destination, file)):
                                print(f"Moving {os.path.join(root, file)} to {destination}")
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"Skipping {file} because it already exists in {destination}")

    # 修改参数
    def modify_workflow(
        self,
        wf_dict: dict,
        prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        strength_model: Optional[float] = None,
        sampler_name: Optional[str] = None,
        denoise: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        """
        修改 ComfyUI workflow 中的关键参数，兼容：
        - ComfyUI API 格式：{ "78": {"class_type": "...","inputs": {...}}, ... }
        - 旧的 "nodes" 数组格式：{ "nodes": [ {"parameters": {...}}, ... ] }
        """
        # 旧格式（保持向后兼容）
        if "nodes" in wf_dict:
            for node in wf_dict["nodes"]:
                params = node.get("parameters", {})
                if prompt and "prompt" in params:
                    params["prompt"] = prompt
                if steps is not None and "steps" in params:
                    params["steps"] = steps
                if seed is not None and "seed" in params:
                    params["seed"] = seed
                node["parameters"] = params
            return wf_dict

        # ComfyUI API 格式
        for node_id, node in wf_dict.items():
            ctype = node.get("class_type")
            inputs = node.get("inputs", {})

            # 正向提示词
            if ctype == "CLIPTextEncode" and prompt:
                if "text" in inputs:
                    inputs["text"] = prompt

            # 分辨率 & 批量
            if ctype == "EmptyLatentImage":
                if width is not None:
                    inputs["width"] = width
                if height is not None:
                    inputs["height"] = height
                if batch_size is not None:
                    inputs["batch_size"] = batch_size

            # 步数 & denoise
            if ctype == "BasicScheduler":
                if steps is not None:
                    inputs["steps"] = steps
                if denoise is not None:
                    inputs["denoise"] = denoise

            # 随机种子
            if ctype == "RandomNoise" and seed is not None:
                inputs["noise_seed"] = seed

            # LoRA 强度
            if ctype == "LoraLoaderModelOnly" and strength_model is not None:
                inputs["strength_model"] = strength_model

            # 采样器
            if ctype == "KSamplerSelect" and sampler_name:
                inputs["sampler_name"] = sampler_name

            node["inputs"] = inputs

        return wf_dict

    def predict(
        self,
        workflow_json: str = Input(
            description="ComfyUI工作流的JSON字符串或URL。使用ComfyUI的'保存（API格式）'功能。",
            default="",
        ),
        prompt: str = Input(description="覆盖工作流中的提示词", default=""),
        width: int = Input(description="输出图像宽度", default=512),
        height: int = Input(description="输出图像高度", default=512),
        steps: int = Input(description="采样步骤数", default=20),
        seed: int = Input(description="随机种子（0表示随机）", default=0),
        strength_model: float = Input(description="LoRA强度（0.0-1.0）", default=1.0),
        sampler_name: str = Input(description="采样算法（例如euler、euler_a、dpmpp_2m）", default=""),
        denoise: float = Input(description="去噪强度（0.0-1.0）", default=1.0),
        batch_size: int = Input(description="批量大小", default=1),
        return_temp_files: bool = Input(description="是否返回临时文件", default=False),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        randomise_seeds: bool = Input(description="自动随机化种子", default=True),
        force_reset_cache: bool = Input(description="强制重置ComfyUI缓存", default=False),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 读取/下载 workflow JSON
        workflow_json_content = workflow_json
        if workflow_json.startswith(("http://", "https://")):
            try:
                response = requests.get(workflow_json)
                response.raise_for_status()
                workflow_json_content = response.text
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download workflow JSON from URL: {e}")

        if not workflow_json_content:
            workflow_json_content = EXAMPLE_WORKFLOW_JSON

        # 修改 workflow 参数（含 prompt 修正）
        wf_dict = json.loads(workflow_json_content)
        wf_dict = self.modify_workflow(
            wf_dict,
            prompt=prompt if prompt else None,
            width=width,
            height=height,
            steps=steps,
            seed=None if (seed == 0 and randomise_seeds) else seed,
            strength_model=strength_model,
            sampler_name=sampler_name if sampler_name else None,
            denoise=denoise,
            batch_size=batch_size,
        )
        workflow_json_content = json.dumps(wf_dict)

        # 加载并运行
        wf = self.comfyUI.load_workflow(workflow_json_content)
        self.comfyUI.connect()

        if force_reset_cache or not randomise_seeds:
            self.comfyUI.reset_execution_cache()

        if randomise_seeds and seed == 0:
            self.comfyUI.randomise_seeds(wf)

        self.comfyUI.run_workflow(wf)

        # 收集输出并做优化
        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(output_directories)
        )
        return [Path(p) for p in optimised_files]
