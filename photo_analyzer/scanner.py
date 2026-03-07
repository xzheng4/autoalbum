"""
图片扫描器模块
负责扫描图片目录，支持增量处理和断点续传
"""
import os
from pathlib import Path
from typing import List, Generator, Set, Optional
from tqdm import tqdm

from .config import SUPPORTED_FORMATS, PHOTOS_DIR
from .database import Database
from .exif_extractor import EXIFExtractor


class ImageScanner:
    """图片扫描器"""

    def __init__(self, db: Database, photos_dir: str = None):
        self.db = db
        self.photos_dir = Path(photos_dir) if photos_dir else PHOTOS_DIR
        self.exif_extractor = EXIFExtractor()

    def scan_directory(self, recursive: bool = True) -> Generator[str, None, None]:
        """
        扫描目录中的所有图片文件

        Args:
            recursive: 是否递归扫描子目录

        Yields:
            图片文件的绝对路径
        """
        if not self.photos_dir.exists():
            print(f"Photos directory not found: {self.photos_dir}")
            return

        # 收集所有图片文件
        image_files = []

        if recursive:
            # 递归扫描所有子目录
            for pattern in SUPPORTED_FORMATS:
                image_files.extend(self.photos_dir.rglob(f"*{pattern}"))
                # 同时检查大写扩展名
                image_files.extend(self.photos_dir.rglob(f"*{pattern.upper()}"))
        else:
            # 只扫描当前目录
            for pattern in SUPPORTED_FORMATS:
                image_files.extend(self.photos_dir.glob(f"*{pattern}"))
                image_files.extend(self.photos_dir.glob(f"*{pattern.upper()}"))

        # 去重并转换为字符串
        unique_files = set(str(f) for f in image_files)
        print(f"Found {len(unique_files)} image files in {self.photos_dir}")

        for file_path in unique_files:
            yield file_path

    def get_unprocessed_images(self) -> List[str]:
        """
        获取所有未处理的图片路径

        Returns:
            未处理图片路径列表
        """
        unprocessed = []

        for file_path in self.scan_directory():
            if not self.db.is_processed(file_path):
                unprocessed.append(file_path)

        print(f"Found {len(unprocessed)} unprocessed images")
        return unprocessed

    def get_new_images(self) -> List[dict]:
        """
        获取数据库中不存在的新图片

        Returns:
            新图片信息列表
        """
        new_images = []

        for file_path in self.scan_directory():
            existing = self.db.get_image_by_path(file_path)
            if not existing:
                # 获取文件基本信息
                try:
                    stat = os.stat(file_path)
                    new_images.append({
                        "file_path": file_path,
                        "file_size": stat.st_size,
                    })
                except Exception as e:
                    print(f"Error getting stats for {file_path}: {e}")

        print(f"Found {len(new_images)} new images")
        return new_images

    def add_new_images_to_db(self) -> int:
        """
        将新图片添加到数据库（但未处理）

        Returns:
            添加的图片数量
        """
        # 先获取所有新图片
        new_images = self.get_new_images()

        if not new_images:
            print("No new images to add")
            return 0

        count = 0

        for image_info in tqdm(new_images, desc="Adding images"):
            try:
                # 计算文件哈希
                file_hash = self.exif_extractor.get_file_hash(image_info["file_path"])

                # 添加到数据库
                self.db.add_image(
                    file_path=image_info["file_path"],
                    file_hash=file_hash,
                    file_size=image_info["file_size"],
                )
                count += 1

            except Exception as e:
                print(f"Error adding {image_info['file_path']}: {e}")

        print(f"Added {count} new images to database")
        return count

    def get_status(self) -> dict:
        """
        获取扫描状态

        Returns:
            包含统计信息的字典
        """
        total_images = sum(1 for _ in self.scan_directory())
        db_count = self.db.get_image_count()
        processed_count = self.db.get_processed_count()

        return {
            "total_files_on_disk": total_images,
            "total_in_database": db_count,
            "processed": processed_count,
            "unprocessed": db_count - processed_count,
            "not_in_database": total_images - db_count,
        }

    def full_scan(self) -> dict:
        """
        执行完整扫描：添加新图片到数据库

        Returns:
            扫描统计信息
        """
        print("Starting full scan...")

        # 添加新图片
        added = self.add_new_images_to_db()

        # 获取状态
        status = self.get_status()
        status["newly_added"] = added

        return status
