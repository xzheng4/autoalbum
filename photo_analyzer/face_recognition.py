"""
人脸识别模块
基于 insightface 库的 buffalo_l 模型
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import cv2
from insightface.app import FaceAnalysis

from .config import ANALYZER_CONFIG, FACES_DIR


class FaceRecognizer:
    """人脸识别器"""

    def __init__(self):
        self.known_face_names = []
        self.known_face_embeddings = []
        self.tolerance = ANALYZER_CONFIG.get("face_tolerance", 0.6)

        # 初始化 insightface FaceAnalysis 模型
        self.face_analyzer = None
        self._init_face_model()

        # 每次启动时自动从 data/faces/ 目录刷新已知人脸数据
        self._refresh_known_faces_from_directory()

    def _init_face_model(self):
        """初始化人脸检测和识别模型"""
        model_name = ANALYZER_CONFIG.get("face_detection_model", "buffalo_l")
        print(f"Initializing face recognition model: {model_name}")

        try:
            # 使用 insightface 的 FaceAnalysis
            self.face_analyzer = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Face recognition model '{model_name}' initialized successfully")
        except Exception as e:
            print(f"Error initializing face recognition model: {e}")
            # 如果 buffalo_l 不可用，尝试 buffalo_s（更小更快）
            try:
                print("Trying buffalo_s model instead...")
                self.face_analyzer = FaceAnalysis(
                    name="buffalo_s",
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                print("Face recognition model 'buffalo_s' initialized successfully")
            except Exception as e2:
                print(f"Error initializing buffalo_s: {e2}")
                self.face_analyzer = None

    def _refresh_known_faces_from_directory(self):
        """
        从 data/faces/ 目录刷新已知人脸数据
        每次启动时自动执行，确保人脸数据是最新的
        目录结构：faces/张三/photo1.jpg, faces/李四/photo2.jpg
        """
        self.known_face_names = []
        self.known_face_embeddings = []

        if not FACES_DIR.exists():
            print(f"Faces directory not found: {FACES_DIR}")
            return

        if not self.face_analyzer:
            print("Face analyzer not initialized, skipping face refresh")
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
                        # 使用 OpenCV 加载图片
                        img = cv2.imread(str(image_path))
                        if img is None:
                            print(f"  Failed to load image: {image_path}")
                            continue

                        # 检测人脸
                        faces = self.face_analyzer.get(img)

                        if faces:
                            # 取第一张检测到的人脸
                            for face in faces:
                                self.known_face_embeddings.append(face.embedding)
                                self.known_face_names.append(person_name)
                                person_faces += 1
                                faces_count += 1
                                print(f"  Found face in {image_path}")
                        else:
                            print(f"  No face detected in {image_path}")

                    except Exception as e:
                        print(f"  Error processing {image_path}: {e}")

                if person_faces > 0:
                    print(f"  Loaded {person_faces} face(s) for {person_name}")

        print(f"Loaded {len(self.known_face_names)} known faces for {len(set(self.known_face_names))} person(s)")

    def register_person(self, name: str, image_paths: List[str]) -> bool:
        """
        注册家庭成员人脸（动态添加，不保存到缓存）

        Args:
            name: 人名
            image_paths: 该人的照片路径列表

        Returns:
            bool: 是否注册成功
        """
        if not self.face_analyzer:
            print("Face analyzer not initialized")
            return False

        embeddings = []

        for image_path in image_paths:
            try:
                # 使用 OpenCV 加载图片
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # 检测人脸
                faces = self.face_analyzer.get(img)

                if faces:
                    # 取第一张检测到的人脸
                    embeddings.append(faces[0].embedding)
                    print(f"Found face in {image_path}")
                else:
                    print(f"No face detected in {image_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if embeddings:
            # 添加新成员的人脸 embedding
            self.known_face_embeddings.extend(embeddings)
            self.known_face_names.extend([name] * len(embeddings))
            print(f"Registered {name} with {len(embeddings)} face(s)")
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

        if not self.face_analyzer:
            print("Face analyzer not initialized")
            return results

        try:
            # 使用 OpenCV 加载图片
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to load image: {image_path}")
                return results

            # 检测人脸
            faces = self.face_analyzer.get(img)

            if not faces:
                return results

            for face in faces:
                result = {
                    "bbox": None,
                    "person_name": None,
                    "confidence": 0.0,
                    "face_embedding": face.embedding.tobytes()
                }

                # bbox: [x1, y1, x2, y2] -> (x, y, w, h)
                bbox = face.bbox.astype(int)
                result["bbox"] = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2] - bbox[0]),
                    int(bbox[3] - bbox[1])
                )

                # 与已知人脸匹配
                if self.known_face_embeddings:
                    # 计算与所有已知人脸的余弦相似度
                    similarities = []
                    for known_embedding in self.known_face_embeddings:
                        sim = self._cosine_similarity(face.embedding, known_embedding)
                        similarities.append(sim)

                    # 找到最佳匹配
                    best_match_idx = int(np.argmax(similarities))
                    best_similarity = similarities[best_match_idx]

                    if best_similarity >= self.tolerance:
                        result["person_name"] = self.known_face_names[best_match_idx]
                        result["confidence"] = float(best_similarity)
                    else:
                        result["person_name"] = "未知"
                        result["confidence"] = float(best_similarity)
                else:
                    result["person_name"] = "未知"

                results.append(result)

        except Exception as e:
            print(f"Error recognizing faces in {image_path}: {e}")

        return results

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个 embedding 的余弦相似度"""
        # 归一化
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)

        # 余弦相似度
        similarity = np.dot(emb1_norm, emb2_norm)

        # 映射到 0-1 范围 (原始范围约 -1 到 1)
        return (similarity + 1) / 2

    def get_registered_persons(self) -> List[str]:
        """获取所有已注册的人物名单"""
        return list(set(self.known_face_names))
