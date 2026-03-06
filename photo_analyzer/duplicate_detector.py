"""
重复照片检测模块
使用感知哈希 (pHash) 算法
"""
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class DuplicateDetector:
    """重复照片检测器"""

    def __init__(self, hash_size: int = 8, mean_diff_threshold: int = 5):
        """
        初始化检测器

        Args:
            hash_size: 哈希大小，越大越精确但越慢
            mean_diff_threshold: 感知哈希的平均差异阈值
        """
        self.hash_size = hash_size
        self.mean_diff_threshold = mean_diff_threshold

    def compute_phash(self, image_path: str) -> Optional[str]:
        """
        计算图片的感知哈希

        Args:
            image_path: 图片路径

        Returns:
            str: 感知哈希字符串
        """
        try:
            with Image.open(image_path) as img:
                # 计算感知哈希
                phash = imagehash.phash(img, hash_size=self.hash_size)
                return str(phash)
        except Exception as e:
            print(f"Error computing phash for {image_path}: {e}")
            return None

    def compute_hashes_for_images(self, image_paths: List[str]) -> Dict[str, str]:
        """
        为多张图片计算感知哈希

        Args:
            image_paths: 图片路径列表

        Returns:
            dict: {file_path: phash}
        """
        results = {}
        for i, path in enumerate(image_paths):
            phash = self.compute_phash(path)
            if phash:
                results[path] = phash

            if (i + 1) % 100 == 0:
                print(f"Computed hashes for {i + 1}/{len(image_paths)} images")

        return results

    def find_duplicates(self, image_hashes: Dict[str, str]) -> List[Tuple[str, str, float]]:
        """
        查找重复图片

        Args:
            image_hashes: {file_path: phash} 字典

        Returns:
            list: [(path1, path2, similarity), ...] 重复对列表
        """
        duplicates = []

        # 按哈希值分组
        hash_groups = defaultdict(list)
        for path, phash in image_hashes.items():
            hash_groups[phash].append(path)

        # 完全相同的哈希值
        for phash, paths in hash_groups.items():
            if len(paths) > 1:
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        duplicates.append((paths[i], paths[j], 1.0))

        # 相似的哈希值（使用汉明距离）
        hash_list = list(hash_groups.keys())
        for i in range(len(hash_list)):
            for j in range(i + 1, len(hash_list)):
                hash1, hash2 = hash_list[i], hash_list[j]
                try:
                    # 计算汉明距离
                    distance = imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
                    # 汉明距离越小越相似
                    if distance <= self.mean_diff_threshold:
                        similarity = 1.0 - (distance / (self.hash_size * self.hash_size))
                        for path1 in hash_groups[hash1]:
                            for path2 in hash_groups[hash2]:
                                duplicates.append((path1, path2, similarity))
                except Exception as e:
                    print(f"Error comparing hashes: {e}")

        return duplicates

    def find_duplicates_for_new_image(self, new_image_path: str,
                                       existing_hashes: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        检查新图片是否与已有图片重复

        Args:
            new_image_path: 新图片路径
            existing_hashes: 已有图片的哈希字典

        Returns:
            list: [(existing_path, similarity), ...]
        """
        results = []

        # 计算新图片的哈希
        new_hash = self.compute_phash(new_image_path)
        if not new_hash:
            return results

        # 与已有图片比较
        for existing_path, existing_hash in existing_hashes.items():
            try:
                # 完全匹配
                if new_hash == existing_hash:
                    results.append((existing_path, 1.0))
                    continue

                # 相似匹配
                distance = imagehash.hex_to_hash(new_hash) - imagehash.hex_to_hash(existing_hash)
                if distance <= self.mean_diff_threshold:
                    similarity = 1.0 - (distance / (self.hash_size * self.hash_size))
                    results.append((existing_path, similarity))

            except Exception as e:
                print(f"Error comparing with {existing_path}: {e}")

        return results

    def get_image_hash(self, image_path: str) -> Optional[str]:
        """获取单个图片的哈希值（供外部调用）"""
        return self.compute_phash(image_path)
