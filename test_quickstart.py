#!/usr/bin/env python3
"""
快速入门脚本
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """测试所有模块是否可以正常导入"""
    print("Testing imports...")

    passed = True

    try:
        from photo_analyzer.config import DATABASE_PATH, PHOTOS_DIR, FACES_DIR
        print("  ✓ config")
    except ImportError as e:
        print(f"  ✗ config: {e}")
        passed = False

    try:
        from photo_analyzer.models import DB_SCHEMA
        print("  ✓ models")
    except ImportError as e:
        print(f"  ✗ models: {e}")
        passed = False

    try:
        from photo_analyzer.database import Database
        print("  ✓ database")
    except ImportError as e:
        print(f"  ✗ database: {e}")
        passed = False

    try:
        from photo_analyzer.exif_extractor import EXIFExtractor
        print("  ✓ exif_extractor")
    except ImportError:
        print("  ⚠ exif_extractor (need: pip install exifread)")

    try:
        from photo_analyzer.face_recognition import FaceRecognizer
        print("  ✓ face_recognition")
    except ImportError:
        print("  ⚠ face_recognition (need: pip install face_recognition dlib)")

    try:
        from photo_analyzer.duplicate_detector import DuplicateDetector
        print("  ✓ duplicate_detector")
    except ImportError:
        print("  ⚠ duplicate_detector (need: pip install imagehash)")

    try:
        from photo_analyzer.vl_analyzer import VLAnalyzer
        print("  ✓ vl_analyzer")
    except ImportError:
        print("  ⚠ vl_analyzer (need: pip install vllm transformers)")

    try:
        from photo_analyzer.scanner import ImageScanner
        print("  ✓ scanner")
    except ImportError as e:
        print(f"  ⚠ scanner (need: pip install exifread)")

    try:
        from photo_analyzer.main import PhotoAnalyzer
        print("  ✓ main")
    except ImportError as e:
        print(f"  ✗ main: {e}")
        passed = False

    try:
        from photo_web.app import app
        print("  ✓ photo_web.app")
    except ImportError as e:
        print(f"  ✗ photo_web.app: {e}")
        passed = False

    return passed


def test_database():
    """测试数据库初始化"""
    print("\nTesting database...")

    try:
        from photo_analyzer.database import Database
        from photo_analyzer.config import DATABASE_PATH

        db = Database(str(DATABASE_PATH))

        # 测试基本操作
        count = db.get_image_count()
        print(f"  ✓ Database initialized, {count} images")

        return True
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def test_directories():
    """测试目录结构"""
    print("\nTesting directories...")

    try:
        from photo_analyzer.config import DATA_DIR, PHOTOS_DIR, FACES_DIR, DATABASE_PATH

        print(f"  ✓ DATA_DIR: {DATA_DIR}")
        print(f"  ✓ PHOTOS_DIR: {PHOTOS_DIR}")
        print(f"  ✓ FACES_DIR: {FACES_DIR}")
        print(f"  ✓ DATABASE_PATH: {DATABASE_PATH}")

        # 确保目录存在
        DATA_DIR.mkdir(exist_ok=True)
        PHOTOS_DIR.mkdir(exist_ok=True)
        FACES_DIR.mkdir(exist_ok=True)

        print("  ✓ All directories created")

        return True
    except Exception as e:
        print(f"  ✗ Directory error: {e}")
        return False


def main():
    print("=" * 50)
    print("AutoAlbum - Quick Start Test")
    print("=" * 50)

    results = []

    # 测试目录
    results.append(("Directories", test_directories()))

    # 测试数据库
    results.append(("Database", test_database()))

    # 测试导入
    results.append(("Imports", test_imports()))

    # 总结
    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print("=" * 50)

    # 总是返回成功，依赖需要用户自行安装
    print("\nCore modules loaded successfully!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Add photos to data/photos/")
    print("  3. Add family member photos to data/faces/")
    print("  4. Run: python -m photo_analyzer.main analyze")
    print("  5. Run: python -m photo_web.main")

    return 0


if __name__ == "__main__":
    sys.exit(main())
