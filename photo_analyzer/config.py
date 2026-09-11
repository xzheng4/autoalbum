"""
AutoAlbum 配置管理
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
DATABASE_PATH = DATA_DIR / "autoalbum.db"

# 确保基础目录存在
DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

# 照片目录 - 可能是 NFS 挂载点，不强制创建
# 支持通过环境变量 PHOTOS_DIR 覆盖
photos_dir_env = os.environ.get("PHOTOS_DIR")
if photos_dir_env:
    PHOTOS_DIR = Path(photos_dir_env)

# 默认 NFS 路径
_default_photos_dir = Path("/mnt/nfs/photos")

# 检查目录是否可访问
def _check_photos_dir(path: Path) -> bool:
    """检查照片目录是否可访问"""
    try:
        # 检查目录是否存在
        if not path.exists():
            return False
        # 尝试列出目录内容来验证可访问性
        list(path.iterdir())
        return True
    except (OSError, PermissionError):
        return False

# 确定最终的照片目录
if 'PHOTOS_DIR' not in locals() or not _check_photos_dir(PHOTOS_DIR):
    if _check_photos_dir(_default_photos_dir):
        PHOTOS_DIR = _default_photos_dir
    else:
        # NFS 不可访问时，使用本地备用目录
        print(f"Warning: /mnt/nfs/photos not accessible")
        PHOTOS_DIR = DATA_DIR / "photos"
        PHOTOS_DIR.mkdir(exist_ok=True)
        print(f"Using fallback photos directory: {PHOTOS_DIR}")

# 照片分析配置
ANALYZER_CONFIG = {
    "batch_size": 4,  # vLLM 批处理大小：1/2/4/8/16/32
    "max_image_size": 2048,  # 分析前图片最大边长
    "face_detection_model": "buffalo_l",  # insightface 模型：'buffalo_l' / 'buffalo_s'
    "face_tolerance": 0.6,  # 人脸识别相似度阈值
}

# vLLM 配置
VLLM_CONFIG = {
    "model_name": "/mnt/d/models/Qwen3-VL-4B-Instruct-FP8",  # 本地 Qwen 模型绝对路径
    "tensor_parallel_size": 1,  # GPU 数量
    "max_model_len": 32768,  # 最大上下文长度 32k
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
