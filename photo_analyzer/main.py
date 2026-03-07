"""
AutoAlbum 照片分析主入口
"""
import argparse
from tqdm import tqdm

from .config import DATABASE_PATH, PHOTOS_DIR
from .database import Database
from .scanner import ImageScanner
from .exif_extractor import EXIFExtractor
from .face_recognition import FaceRecognizer
from .vl_analyzer import VLAnalyzer
from .duplicate_detector import DuplicateDetector


class PhotoAnalyzer:
    """照片分析器主类"""

    def __init__(self, db_path: str = None, batch_size: int = None, photos_dir: str = None):
        self.db_path = db_path or str(DATABASE_PATH)
        self.photos_dir = photos_dir
        self.db = Database(self.db_path)
        self.scanner = ImageScanner(self.db, photos_dir=photos_dir)
        self.exif_extractor = EXIFExtractor()
        self.face_recognizer = FaceRecognizer()
        self.vl_analyzer = VLAnalyzer(batch_size=batch_size)
        self.duplicate_detector = DuplicateDetector()

    def register_faces_from_directory(self) -> dict:
        """从 faces 目录刷新家庭成员人脸"""
        print("=" * 50)
        print("Refreshing known faces from data/faces/ directory...")
        print("=" * 50)

        # FaceRecognizer 会在初始化时自动刷新
        self.face_recognizer = FaceRecognizer()

        persons = self.face_recognizer.get_registered_persons()
        print(f"\nRegistered persons: {len(persons)}")
        for name in persons:
            print(f"  - {name}")

        return {name: True for name in persons}

    def scan(self) -> dict:
        """扫描图片目录，添加新图片到数据库"""
        print("=" * 50)
        print("Scanning photos directory...")
        print("=" * 50)

        status = self.scanner.full_scan()

        print(f"  Total files on disk: {status['total_files_on_disk']}")
        print(f"  Total in database: {status['total_in_database']}")
        print(f"  Processed: {status['processed']}")
        print(f"  Unprocessed: {status['unprocessed']}")
        print(f"  Not in database: {status['not_in_database']}")

        return status

    def get_status(self) -> dict:
        """获取当前状态"""
        return self.scanner.get_status()

    def refresh_exif(self, limit: int = None) -> dict:
        """
        强制重新刷新所有图片的 EXIF 信息

        Args:
            limit: 限制处理的图片数量

        Returns:
            dict: 处理统计信息
        """
        print("=" * 50)
        print("Refreshing EXIF data for all images...")
        print("=" * 50)

        # 获取所有已处理的图片
        all_images = self.db.get_all_images(limit=limit)

        if not all_images:
            print("No images found in database!")
            return {"total": 0, "success": 0, "failed": 0, "updated": 0}

        print(f"Found {len(all_images)} images to process")

        # 批量处理
        stats = {"total": len(all_images), "success": 0, "failed": 0, "updated": 0}

        for image in tqdm(all_images, desc="Refreshing EXIF"):
            image_path = image['file_path']
            image_id = image['id']

            try:
                # 提取 EXIF 信息
                exif_data = self.exif_extractor.extract_exif(image_path)

                # 更新图片基本信息
                self.db.add_image(
                    file_path=image_path,
                    file_hash=exif_data.get('file_hash') or self.exif_extractor.get_file_hash(image_path),
                    file_size=exif_data.get('file_size') or self.exif_extractor.get_file_size(image_path),
                    width=exif_data.get('width'),
                    height=exif_data.get('height'),
                    format=exif_data.get('format'),
                    captured_at=exif_data.get('captured_at'),
                )

                # 更新 EXIF 数据
                self.db.add_exif_data(image_id, **exif_data)

                stats["success"] += 1
                stats["updated"] += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                stats["failed"] += 1

        # 打印总结
        print("\n" + "=" * 50)
        print("EXIF Refresh Complete!")
        print("=" * 50)
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  EXIF records updated: {stats['updated']}")

        return stats

    def refresh_faces(self, limit: int = None, batch_size: int = None) -> dict:
        """
        强制重新刷新所有图片的人脸信息（保留 VL 分析内容不变）

        Args:
            limit: 限制处理的图片数量
            batch_size: 批处理大小

        Returns:
            dict: 处理统计信息
        """
        print("=" * 50)
        print("Refreshing face recognition for all images...")
        print("(EXIF and VL analysis data will be preserved)")
        print("=" * 50)

        batch_size = batch_size or self.vl_analyzer.batch_size

        # 获取所有已处理的图片
        all_images = self.db.get_all_images(limit=limit)

        if not all_images:
            print("No images found in database!")
            return {"total": 0, "success": 0, "failed": 0, "updated": 0}

        print(f"Found {len(all_images)} images to process")

        # 批量处理
        stats = {"total": len(all_images), "success": 0, "failed": 0, "updated": 0}

        batches = [all_images[i:i + batch_size] for i in range(0, len(all_images), batch_size)]

        for batch in tqdm(batches, desc="Refreshing faces"):
            for image in batch:
                image_path = image['file_path']
                image_id = image['id']

                try:
                    # 删除旧的人脸数据
                    with self.db.get_connection() as conn:
                        conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
                        conn.commit()

                    # 重新识别人脸
                    faces = self.face_recognizer.recognize_faces(image_path)

                    # 保存新的人脸数据
                    for face in faces:
                        if face.get('face_embedding'):
                            self.db.add_face(
                                image_id=image_id,
                                person_name=face.get('person_name', '未知'),
                                face_encoding=face.get('face_embedding'),
                                bbox=face.get('bbox', (0, 0, 0, 0)),
                                confidence=face.get('confidence'),
                            )
                            stats["updated"] += 1

                    stats["success"] += 1

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    stats["failed"] += 1

        # 打印总结
        print("\n" + "=" * 50)
        print("Face Refresh Complete!")
        print("=" * 50)
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Face records updated: {stats['updated']}")

        # 显示家庭成员统计
        persons = self.db.get_all_persons()
        print(f"\nRegistered persons: {len(persons)}")
        for name in persons:
            images_with_person = self.db.get_images_by_person(name)
            print(f"  - {name}: {len(images_with_person)} photos")

        return stats

    def refresh_vl_analysis(self, limit: int = None, batch_size: int = None) -> dict:
        """
        强制重新刷新所有图片的 VL 分析信息（保留人脸信息不变）

        Args:
            limit: 限制处理的图片数量
            batch_size: 批处理大小

        Returns:
            dict: 处理统计信息
        """
        print("=" * 50)
        print("Refreshing VL analysis for all images...")
        print("(EXIF and Face recognition data will be preserved)")
        print("=" * 50)

        batch_size = batch_size or self.vl_analyzer.batch_size

        # 获取所有已处理的图片
        all_images = self.db.get_all_images(limit=limit)

        if not all_images:
            print("No images found in database!")
            return {"total": 0, "success": 0, "failed": 0, "updated": 0}

        print(f"Found {len(all_images)} images to process")
        print(f"Batch size: {batch_size}")

        # 批量处理
        stats = {"total": len(all_images), "success": 0, "failed": 0, "updated": 0}

        batches = [all_images[i:i + batch_size] for i in range(0, len(all_images), batch_size)]

        for batch_num, batch in enumerate(tqdm(batches, desc="Refreshing VL analysis"), 1):
            for image in batch:
                image_path = image['file_path']
                image_id = image['id']

                try:
                    # 删除旧的 VL 分析数据
                    with self.db.get_connection() as conn:
                        conn.execute("DELETE FROM vl_analysis WHERE image_id = ?", (image_id,))
                        conn.commit()

                    # 重新进行 VL 分析
                    vl_result = self.vl_analyzer.analyze_image(image_path)

                    # 保存新的 VL 分析数据
                    if vl_result:
                        self.db.add_vl_analysis(
                            image_id=image_id,
                            ocr_text=vl_result.get('ocr_text'),
                            scene_description=vl_result.get('scene_description'),
                            category=vl_result.get('category'),
                            objects=vl_result.get('objects'),
                            mood=vl_result.get('mood'),
                            confidence=vl_result.get('confidence'),
                        )
                        stats["updated"] += 1

                    stats["success"] += 1

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    stats["failed"] += 1

        # 打印总结
        print("\n" + "=" * 50)
        print("VL Analysis Refresh Complete!")
        print("=" * 50)
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  VL records updated: {stats['updated']}")

        return stats

    def analyze_all(self, limit: int = None) -> dict:
        """
        分析所有图片：依次执行 refresh-exif, refresh-faces, refresh-vl

        Args:
            limit: 限制处理的图片数量

        Returns:
            dict: 分析统计信息
        """
        print("=" * 50)
        print("Starting full photo analysis pipeline...")
        print("=" * 50)
        print(f"Batch size: {self.vl_analyzer.batch_size}")
        print()

        # 1. 刷新 EXIF
        print("\n>>> Step 1/3: Refreshing EXIF data...")
        self.refresh_exif(limit=limit)

        # 2. 刷新人脸
        print("\n>>> Step 2/3: Refreshing face recognition...")
        self.refresh_faces(limit=limit)

        # 3. 刷新 VL 分析
        print("\n>>> Step 3/3: Refreshing VL analysis...")
        self.refresh_vl_analysis(limit=limit)

        # 打印总结
        print("\n" + "=" * 50)
        print("Full Analysis Pipeline Complete!")
        print("=" * 50)

        return {"status": "complete"}

    def close(self):
        """清理资源"""
        self.vl_analyzer.close()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="AutoAlbum Photo Analyzer")
    parser.add_argument("command",
                        choices=["scan", "analyze", "register", "status", "refresh-exif", "refresh-faces", "refresh-vl"],
                        help="Command to run")
    parser.add_argument("--db", type=str, help="Database path")
    parser.add_argument("--batch-size", type=int, default=4,
                        choices=[1, 2, 4, 8, 16, 32],
                        help="Batch size for vLLM inference")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--photos-dir", type=str, help="Photos directory path")

    args = parser.parse_args()

    # 创建分析器
    analyzer = PhotoAnalyzer(
        db_path=args.db,
        batch_size=args.batch_size,
        photos_dir=args.photos_dir,
    )

    try:
        if args.command == "scan":
            analyzer.scan()

        elif args.command == "analyze":
            # 先扫描添加新图片
            analyzer.scan()
            # 然后执行完整分析流程
            analyzer.analyze_all(limit=args.limit)

        elif args.command == "register":
            analyzer.register_faces_from_directory()

        elif args.command == "status":
            status = analyzer.get_status()
            print("\nAutoAlbum Status:")
            print("-" * 30)
            for key, value in status.items():
                print(f"  {key}: {value}")

        elif args.command == "refresh-exif":
            # 强制重新刷新 EXIF 信息
            analyzer.refresh_exif(limit=args.limit)

        elif args.command == "refresh-faces":
            # 强制重新刷新人脸信息
            analyzer.refresh_faces(limit=args.limit, batch_size=args.batch_size)

        elif args.command == "refresh-vl":
            # 强制重新刷新 VL 分析信息
            analyzer.refresh_vl_analysis(limit=args.limit, batch_size=args.batch_size)

    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
