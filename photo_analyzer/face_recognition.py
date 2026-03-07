"""
人脸识别模块
基于 face_recognition 库（dlib）
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import face_recognition

from .config import ANALYZER_CONFIG, FACES_DIR


class FaceRecognizer:
    """人脸识别器"""

    def __init__(self, known_faces_file: str = None):
        self.known_faces_file = known_faces_file or str(Path(FACES_DIR) / "known_faces.pkl")
        self.known_face_names = []
        self.known_face_encodings = []
        self.model = ANALYZER_CONFIG.get("face_detection_model", "hog")
        self.tolerance = ANALYZER_CONFIG.get("face_tolerance", 0.6)

        # 每次启动时自动从 data/faces/ 目录刷新已知人脸数据
        self._refresh_known_faces_from_directory()

    def _refresh_known_faces_from_directory(self):
        """
        从 data/faces/ 目录刷新已知人脸数据
        每次启动时自动执行，确保人脸数据是最新的
        目录结构：faces/张三/photo1.jpg, faces/李四/photo2.jpg
        """
        self.known_face_names = []
        self.known_face_encodings = []

        if not FACES_DIR.exists():
            print(f"Faces directory not found: {FACES_DIR}")
            return

        print("Refreshing known faces from data/faces/ directory...")
        faces_count = 0

        # 遍历 faces 目录下的子目录（每个人名）
        for person_dir in FACES_DIR.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                person_faces = 0

                # 收集该人的所有照片
                image_paths = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                    image_paths.extend(person_dir.glob(ext))

                # 处理每张照片
                for image_path in image_paths:
                    try:
                        # 加载图片
                        image = face_recognition.load_image_file(str(image_path))

                        # 检测人脸
                        face_encodings = face_recognition.face_encodings(image)

                        if face_encodings:
                            # 取第一张检测到的人脸
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(person_name)
                            person_faces += 1
                            faces_count += 1
                        else:
                            print(f"  No face detected in {image_path}")

                    except Exception as e:
                        print(f"  Error processing {image_path}: {e}")

                if person_faces > 0:
                    print(f"  Loaded {person_faces} face(s) for {person_name}")

        # 保存缓存（可选，用于加速下次启动）
        if faces_count > 0:
            self._save_known_faces()

        print(f"Loaded {len(self.known_face_names)} known faces for {len(set(self.known_face_names))} person(s)")

    def _save_known_faces(self):
        """保存已知人脸数据到缓存文件"""
        data = {
            "names": self.known_face_names,
            "encodings": self.known_face_encodings
        }
        with open(self.known_faces_file, "wb") as f:
            pickle.dump(data, f)

    def register_person(self, name: str, image_paths: List[str]) -> bool:
        """
        注册家庭成员人脸（动态添加，不保存到缓存）

        Args:
            name: 人名
            image_paths: 该人的照片路径列表

        Returns:
            bool: 是否注册成功
        """
        encodings = []

        for image_path in image_paths:
            try:
                # 加载图片
                image = face_recognition.load_image_file(image_path)

                # 检测人脸
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    # 取第一张检测到的人脸
                    encodings.append(face_encodings[0])
                    print(f"Found face in {image_path}")
                else:
                    print(f"No face detected in {image_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if encodings:
            # 添加新成员的人脸编码（内存中，不保存到缓存）
            self.known_face_encodings.extend(encodings)
            self.known_face_names.extend([name] * len(encodings))
            print(f"Registered {name} with {len(encodings)} face(s)")
            return True
        else:
            print(f"Could not register {name}: no faces detected")
            return False

    def recognize_faces(self, image_path: str) -> List[Dict]:
        """
        识别图片中的人脸

        Args:
            image_path: 图片路径

        Returns:
            list: 识别结果列表，每个人脸包含 name, bbox, confidence
        """
        results = []

        try:
            # 加载图片
            image = face_recognition.load_image_file(image_path)

            # 检测人脸位置
            face_locations = face_recognition.face_locations(image, model=self.model)

            if not face_locations:
                return results

            # 获取人脸编码
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                result = {
                    "bbox": None,
                    "person_name": None,
                    "confidence": 0.0,
                    "face_encoding": face_encoding.tobytes()
                }

                # 转换 bbox 格式 (top, right, bottom, left) -> (x, y, w, h)
                top, right, bottom, left = face_location
                result["bbox"] = (left, top, right - left, bottom - top)

                # 与已知人脸匹配
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=self.tolerance
                    )

                    # 计算最佳匹配
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                    )

                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        result["person_name"] = self.known_face_names[best_match_idx]
                        # 置信度 = 1 - 距离
                        result["confidence"] = float(1 - face_distances[best_match_idx])
                    else:
                        result["person_name"] = "未知"
                        result["confidence"] = float(1 - np.min(face_distances)) if len(face_distances) > 0 else 0.0
                else:
                    result["person_name"] = "未知"

                results.append(result)

        except Exception as e:
            print(f"Error recognizing faces in {image_path}: {e}")

        return results

    def get_registered_persons(self) -> List[str]:
        """获取所有已注册的人物名单"""
        return list(set(self.known_face_names))
