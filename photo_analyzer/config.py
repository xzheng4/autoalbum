"""
AutoAlbum 配置管理
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PHOTOS_DIR = DATA_DIR / "photos"
FACES_DIR = DATA_DIR / "faces"
DATABASE_PATH = DATA_DIR / "autoalbum.db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

# 照片分析配置
ANALYZER_CONFIG = {
    "batch_size": 4,  # vLLM 批处理大小：1/2/4/8/16/32
    "max_image_size": 2048,  # 分析前图片最大边长
    "face_detection_model": "buffalo_l",  # insightface 模型：'buffalo_l' / 'buffalo_s'
    "face_tolerance": 0.6,  # 人脸识别相似度阈值
}

# vLLM 配置
VLLM_CONFIG = {
    "model_name": "Qwen/Qwen3-VL-4B-Instruct",
    "tensor_parallel_size": 1,  # GPU 数量
    "max_model_len": 4096,  # 最大上下文长度
    "gpu_memory_utilization": 0.8,  # GPU 显存利用率
}

# 支持的图片格式
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

# Web 配置
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "thumbnails_per_page": 48,
}
