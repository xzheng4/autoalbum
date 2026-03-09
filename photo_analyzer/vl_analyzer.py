"""
Qwen3-VL 统一分析模块
使用 vLLM 0.16.0 作为推理引擎，进行 OCR 和图片内容理解
"""
import os
import json
import re
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

**重要限制**：
- `ocr_text` 字段的长度请限制在 1024 tokens 以内。如果图片中的文字内容超过此限制，请截取最重要的前 1024 tokens。

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

        vLLM 0.16.0 的 llm.chat() 支持传入多个对话进行批量处理，
        每个对话是一个完整的消息列表。
        """
        results = [None] * len(image_paths)

        try:
            # 为每个图片构建一个完整的对话
            all_messages = []
            valid_indices = []

            for idx, image_path in enumerate(image_paths):
                try:
                    image_base64 = self._encode_image_to_base64(image_path)
                    # 每个图片是一个独立的对话，包含完整的消息历史
                    conversation = [
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
                    all_messages.append(conversation)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error preparing {image_path}: {e}")

            if not all_messages:
                return results

            # 批量推理 - vLLM 0.16.0 支持传入多个对话进行批量处理
            # 抑制 vLLM 的 "Adding requests" 和 "Processed prompts" 进度输出
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                outputs = self.llm.chat(
                    messages=all_messages,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )
            finally:
                sys.stderr = old_stderr

            # vLLM 返回的输出顺序与输入对话顺序一一对应
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

            # 直接尝试解析（如果模型返回的是有效 JSON）
            result = json.loads(content)
            return self._validate_and_normalize_result(result)

        except Exception as e:
            # 解析失败，尝试各种修复策略
            return self._try_parse_with_fallbacks(content, e)

    def _validate_and_normalize_result(self, result: dict) -> Optional[Dict[str, Any]]:
        """验证并标准化解析结果"""
        return {
            "ocr_text": result.get("ocr_text", ""),
            "scene_description": result.get("scene_description", ""),
            "category": result.get("category", "other"),
            "objects": result.get("objects", []),
            "mood": result.get("mood", ""),
            "confidence": float(result.get("confidence", 0.5)),
        }

    def _try_parse_with_fallbacks(self, content: str, original_error: Exception) -> Optional[Dict[str, Any]]:
        """
        使用多种回退策略解析 JSON

        策略 1: 修复换行符等控制字符后尝试标准解析
        策略 2: 使用健壮的字段级提取（针对 OCR 内容破坏 JSON 的情况）
        策略 3: 提取短字段（category, confidence 等），OCR 文本使用启发式方法
        """
        # 策略 1: 尝试提取 JSON 块并修复控制字符
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_str = self._fix_json_string(json_str)
                result = json.loads(json_str)
                return self._validate_and_normalize_result(result)
        except Exception as e1:
            pass

        # 策略 2: 提取短字段，然后对 ocr_text 使用特殊处理
        try:
            result = self._extract_fields_robust(content)
            if result:
                print(f"Recovered result using robust field extraction")
                return self._validate_and_normalize_result(result)
        except Exception as e2:
            print(f"Robust field extraction failed: {e2}")

        # 所有策略都失败了
        print(f"Error parsing response JSON: {original_error}")
        print(f"Raw content (truncated): {content[:500]}...")
        return None

    def _extract_fields_robust(self, content: str) -> Optional[Dict[str, Any]]:
        """
        健壮的字段提取 - 专门处理 OCR 内容破坏 JSON 结构的情况

        策略：
        1. 首先提取简单的短字段（confidence, category, mood, objects）
        2. 然后找到 "scene_description" 字段的位置
        3. ocr_text 是从 "ocr_text": 到 "scene_description": 之间的内容
        """
        result = {
            "ocr_text": "",
            "scene_description": "",
            "category": "other",
            "objects": [],
            "mood": "",
            "confidence": 0.5,
        }

        # 1. 提取 confidence（数字，最可靠）
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        # 2. 提取 objects（数组）
        objects_match = re.search(r'"objects"\s*:\s*\[([^\]]*)\]', content)
        if objects_match:
            arr_content = objects_match.group(1)
            result["objects"] = re.findall(r'"([^"]*)"', arr_content)

        # 3. 提取 category（短字符串）
        cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', content)
        if cat_match:
            result["category"] = cat_match.group(1)

        # 4. 提取 mood（短字符串）
        mood_match = re.search(r'"mood"\s*:\s*"([^"]+)"', content)
        if mood_match:
            result["mood"] = mood_match.group(1)

        # 5. 找到 scene_description 的起始位置
        scene_match = re.search(r'"scene_description"\s*:\s*"', content)
        scene_start = scene_match.end() if scene_match else None

        if scene_start:
            # 提取 scene_description 的值（找到下一个字段或结尾）
            next_field_patterns = [
                r'"\s*,\s*"category"\s*:',
                r'"\s*,\s*"objects"\s*:',
                r'"\s*,\s*"mood"\s*:',
                r'"\s*,\s*"confidence"\s*:',
            ]
            end_pos = len(content)
            for pattern in next_field_patterns:
                match = re.search(pattern, content)
                if match and match.start() > scene_start and match.start() < end_pos:
                    end_pos = match.start()

            scene_raw = content[scene_start:end_pos].rstrip()
            while scene_raw and scene_raw[-1] in '"\',:':
                scene_raw = scene_raw[:-1].rstrip()

            if scene_raw:
                try:
                    result["scene_description"] = json.loads('"' + scene_raw + '"')
                except:
                    escaped = scene_raw.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    try:
                        result["scene_description"] = json.loads('"' + escaped + '"')
                    except:
                        result["scene_description"] = scene_raw

        # 6. 提取 ocr_text：从 "ocr_text": 到 "scene_description": 之前
        ocr_start_match = re.search(r'"ocr_text"\s*:\s*"', content)
        if ocr_start_match and scene_match:
            start_pos = ocr_start_match.end()
            end_pos = scene_match.start()  # scene_description 字段的开始位置

            ocr_raw = content[start_pos:end_pos].rstrip()

            # 清理末尾（可能是前一个字段的结束标记）
            # 查找最后一个 " 的位置，它应该是前一个字段值的结束
            # 但由于 ocr_text 可能包含未转义的 "，我们需要更聪明的方法

            # 策略：从后往前找第一个后面跟 :\s*"scene 的 "
            # 这通常标志着 scene_description 字段的开始
            # 所以 ocr_raw 的末尾应该清理掉不属于 OCR 内容的部分

            # 简化处理：直接清理末尾的特殊字符
            while ocr_raw and ocr_raw[-1] in '"\\n\\r\\t,':
                ocr_raw = ocr_raw[:-1]

            ocr_raw = ocr_raw.strip()

            # 尝试解析
            if ocr_raw:
                # 首先尝试直接解析
                try:
                    result["ocr_text"] = json.loads('"' + ocr_raw + '"')
                except:
                    # 手动转义
                    escaped = ocr_raw.replace('\\\\', '\\\\\\\\').replace('"', '\\\\"').replace('\\n', '\\\\n').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    try:
                        result["ocr_text"] = json.loads('"' + escaped + '"')
                    except:
                        result["ocr_text"] = ocr_raw

        return result

    def _fix_json_string(self, json_str: str) -> str:
        """
        修复 JSON 字符串中的转义问题

        使用状态机逐字符扫描，正确处理字符串内的特殊字符。
        关键：只转义字符串值内部的特殊字符，不修改 JSON 结构。
        """
        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(json_str):
            char = json_str[i]

            # 处理转义序列
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            # 处理反斜杠（在字符串内）
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                i += 1
                continue

            # 处理双引号（切换字符串状态）
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            # 在字符串内部，需要转义特殊字符
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                else:
                    result.append(char)
            else:
                result.append(char)

            i += 1

        return ''.join(result)

    def close(self):
        """释放资源"""
        if self.llm:
            # vLLM 不需要显式清理
            pass
        self._initialized = False
