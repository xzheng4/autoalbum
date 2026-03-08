"""
Qwen3-VL 统一分析模块
使用 vLLM 0.16.0 作为推理引擎，进行 OCR 和图片内容理解
"""
import os
import json
import base64
from typing import List, Dict, Optional, Any
from pathlib import Path
from PIL import Image
import io

from .config import VLLM_CONFIG, ANALYZER_CONFIG


class VLAnalyzer:
    """
    视觉语言分析器
    使用 Qwen3-VL-4B 模型进行统一的 OCR 提取和图片内容理解
    """

    # 分析用的系统提示和指令
    ANALYSIS_PROMPT = """
你是一位图片分析专家。请分析提供的图片，并输出结构化的 JSON 响应。

对于每张图片，请提供以下信息：
1. **OCR 文字**: 图片中所有可识别的文字（标牌、文档、说明等）
2. **场景描述**: 简要描述图片中的场景或正在发生的事情
3. **类别**: 从以下类别中选择主要类别：[家庭、旅游、美食、聚会、运动、自然、宠物、工作、文档、自拍、合影、室内、户外、活动、其他]
4. **物体**: 列出图片中可见的主要物体/人物
5. **氛围**: 描述图片的整体氛围（快乐、正式、休闲、怀旧等）
6. **置信度**: 你对本次分析的置信度（0.0-1.0）

请仅输出一个有效的 JSON 对象，格式如下：
{
    "ocr_text": "提取的文字内容，如果没有则为空字符串",
    "scene_description": "场景简要描述",
    "category": "类别名称",
    "objects": ["物体 1", "物体 2", ...],
    "mood": "氛围描述",
    "confidence": 0.95
}
"""

    def __init__(self, batch_size: int = None):
        self.batch_size = batch_size or ANALYZER_CONFIG.get("batch_size", 4)
        self.model_name = VLLM_CONFIG.get("model_name", "Qwen/Qwen3-VL-4B-Instruct-FP8")
        self.llm = None
        self._initialized = False

    def initialize(self):
        """初始化 vLLM 模型"""
        if self._initialized:
            return

        print(f"Initializing vLLM with {self.model_name}...")
        try:
            from vllm import LLM, SamplingParams
            from vllm import SamplingParams

            # vLLM 0.16.0 API
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=VLLM_CONFIG.get("tensor_parallel_size", 1),
                max_model_len=VLLM_CONFIG.get("max_model_len", 4096),
                gpu_memory_utilization=VLLM_CONFIG.get("gpu_memory_utilization", 0.8),
                trust_remote_code=True,
            )

            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=2048,
                top_p=0.95,
            )

            self._initialized = True
            print("vLLM initialized successfully")

        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            raise

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为 base64"""
        try:
            with Image.open(image_path) as img:
                # 调整图片大小（如果需要）
                max_size = ANALYZER_CONFIG.get("max_image_size", 2048)
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 转换为 RGB（如果需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 编码为 base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        分析单张图片

        Args:
            image_path: 图片路径

        Returns:
            dict: 分析结果，包含 ocr_text, scene_description, category, objects, mood, confidence
        """
        if not self._initialized:
            self.initialize()

        try:
            # 构建消息
            image_base64 = self._encode_image_to_base64(image_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            # 调用模型
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)

            # 解析结果
            if outputs and len(outputs) > 0:
                content = outputs[0].outputs[0].text
                return self._parse_response(content)

        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")

        return None

    def analyze_batch(self, image_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        批量分析多张图片

        Args:
            image_paths: 图片路径列表

        Returns:
            list: 分析结果列表
        """
        if not self._initialized:
            self.initialize()

        # 直接处理所有图片，由调用者负责分批
        return self._process_batch(image_paths)

    def _process_batch(self, image_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        处理一个 batch 的图片 - 使用 vLLM 批量并行推理

        vLLM 0.16.0 的 llm.chat() 支持传入列表批量处理，
        可以一次性并行处理多个请求，比逐个调用更高效。
        """
        results = [None] * len(image_paths)

        try:
            # 为每个图片构建消息
            messages_list = []
            valid_indices = []

            for idx, image_path in enumerate(image_paths):
                try:
                    image_base64 = self._encode_image_to_base64(image_path)
                    messages = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.ANALYSIS_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                    messages_list.append(messages)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error preparing {image_path}: {e}")

            if not messages_list:
                return results

            # 批量推理 - vLLM 0.16.0 支持传入列表一次性处理多个请求
            # 抑制 vLLM 的 "Adding requests" 和 "Processed prompts" 进度输出
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                outputs = self.llm.chat(
                    messages=messages_list,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
            finally:
                sys.stderr = old_stderr

            # 解析结果并映射回原始索引
            for i, output in enumerate(outputs):
                try:
                    original_idx = valid_indices[i]
                    content = output.outputs[0].text
                    results[original_idx] = self._parse_response(content)
                except Exception as e:
                    print(f"Error parsing result {i}: {e}")

        except Exception as e:
            print(f"Error processing batch: {e}")

        return results

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析模型返回的 JSON 内容"""
        try:
            # 清理 markdown 格式
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 尝试修复常见的 JSON 格式问题
            # 1. 替换未转义的双引号（在字符串值中的）
            # 2. 替换换行符为空格
            content = content.replace('\n', ' ').replace('\r', '')

            # 解析 JSON
            result = json.loads(content)

            # 确保所有字段存在
            return {
                "ocr_text": result.get("ocr_text", ""),
                "scene_description": result.get("scene_description", ""),
                "category": result.get("category", "other"),
                "objects": result.get("objects", []),
                "mood": result.get("mood", ""),
                "confidence": float(result.get("confidence", 0.5)),
            }

        except Exception as e:
            # 尝试使用更宽松的方式解析
            try:
                # 尝试提取 JSON 部分（如果响应包含额外文本）
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # 尝试修复常见的字符串问题
                    json_str = self._fix_json_string(json_str)
                    result = json.loads(json_str)
                    return {
                        "ocr_text": result.get("ocr_text", ""),
                        "scene_description": result.get("scene_description", ""),
                        "category": result.get("category", "other"),
                        "objects": result.get("objects", []),
                        "mood": result.get("mood", ""),
                        "confidence": float(result.get("confidence", 0.5)),
                    }
            except:
                pass

            # JSON 解析失败，返回 None 表示需要跳过该图片
            print(f"Error parsing response JSON: {e}")
            print(f"Raw content: {content[:200]}...")
            return None

    def _fix_json_string(self, json_str: str) -> str:
        """
        修复常见的 JSON 字符串问题

        Args:
            json_str: 可能有问题的 JSON 字符串

        Returns:
            修复后的 JSON 字符串
        """
        # 移除字符串值中的未转义换行符
        in_string = False
        escape_next = False
        result = []

        for char in json_str:
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
            elif char in '\n\r' and in_string:
                # 在字符串中的换行符替换为空格
                result.append(' ')
            else:
                result.append(char)

        return ''.join(result)

    def close(self):
        """释放资源"""
        if self.llm:
            # vLLM 不需要显式清理
            pass
        self._initialized = False
