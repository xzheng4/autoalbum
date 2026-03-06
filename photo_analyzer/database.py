"""
SQLite 数据库操作模块
"""
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .models import DB_SCHEMA


class Database:
    """SQLite 数据库操作类"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # 启用外键
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库 schema"""
        with self.get_connection() as conn:
            conn.executescript(DB_SCHEMA)

    # ==================== Images 表操作 ====================

    def add_image(self, file_path: str, file_hash: str = None,
                  file_size: int = None, width: int = None,
                  height: int = None, format: str = None,
                  captured_at: str = None) -> int:
        """添加或更新图片记录"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO images
                (file_path, file_hash, file_size, width, height, format, captured_at, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)
            """, (file_path, file_hash, file_size, width, height, format, captured_at))
            return cursor.lastrowid

    def get_image_by_path(self, file_path: str) -> Optional[Dict]:
        """根据路径获取图片记录"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM images WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_image_by_id(self, image_id: int) -> Optional[Dict]:
        """根据 ID 获取图片记录"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM images WHERE id = ?", (image_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_unprocessed_images(self, limit: int = None) -> List[Dict]:
        """获取未处理的图片"""
        query = "SELECT * FROM images WHERE processed = FALSE"
        if limit:
            query += f" LIMIT {limit}"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_all_images(self, limit: int = None, offset: int = None) -> List[Dict]:
        """获取所有图片"""
        query = "SELECT * FROM images ORDER BY captured_at DESC, id DESC"
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def mark_processed(self, image_id: int):
        """标记图片为已处理"""
        with self.get_connection() as conn:
            conn.execute("UPDATE images SET processed = TRUE WHERE id = ?", (image_id,))

    def image_exists(self, file_path: str) -> bool:
        """检查图片是否已存在"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM images WHERE file_path = ?", (file_path,)
            )
            return cursor.fetchone() is not None

    def is_processed(self, file_path: str) -> bool:
        """检查图片是否已处理"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT processed FROM images WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()
            return row and row[0]

    def get_image_count(self) -> int:
        """获取图片总数"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM images")
            return cursor.fetchone()[0]

    def get_processed_count(self) -> int:
        """获取已处理图片数量"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM images WHERE processed = TRUE")
            return cursor.fetchone()[0]

    # ==================== EXIF 表操作 ====================

    def add_exif_data(self, image_id: int, **kwargs):
        """添加 EXIF 数据"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO exif_data
                (image_id, make, model, lens_model, iso, aperture,
                 shutter_speed, focal_length, gps_lat, gps_lon, gps_alt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, kwargs.get('make'), kwargs.get('model'),
                  kwargs.get('lens_model'), kwargs.get('iso'),
                  kwargs.get('aperture'), kwargs.get('shutter_speed'),
                  kwargs.get('focal_length'), kwargs.get('gps_lat'),
                  kwargs.get('gps_lon'), kwargs.get('gps_alt')))

    def get_exif_data(self, image_id: int) -> Optional[Dict]:
        """获取 EXIF 数据"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM exif_data WHERE image_id = ?", (image_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== Faces 表操作 ====================

    def add_face(self, image_id: int, person_name: str,
                 face_encoding: bytes, bbox: tuple, confidence: float = None):
        """添加人脸记录"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO faces
                (image_id, person_name, face_encoding, bbox_x, bbox_y, bbox_w, bbox_h, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, person_name, face_encoding,
                  bbox[0], bbox[1], bbox[2], bbox[3], confidence))

    def get_faces_by_image(self, image_id: int) -> List[Dict]:
        """获取图片中的人脸"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM faces WHERE image_id = ?", (image_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_person(self, person_name: str) -> List[Dict]:
        """获取某人的所有图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT i.* FROM images i
                JOIN faces f ON i.id = f.image_id
                WHERE f.person_name = ?
                ORDER BY i.captured_at DESC
            """, (person_name,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_persons(self) -> List[str]:
        """获取所有人物名单"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    # ==================== VL Analysis 表操作 ====================

    def add_vl_analysis(self, image_id: int, ocr_text: str = None,
                        scene_description: str = None, category: str = None,
                        objects: list = None, mood: str = None,
                        confidence: float = None):
        """添加 VL 分析结果"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO vl_analysis
                (image_id, ocr_text, scene_description, category, objects, mood, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_id, ocr_text, scene_description, category,
                  json.dumps(objects) if objects else None, mood, confidence))

    def get_vl_analysis(self, image_id: int) -> Optional[Dict]:
        """获取 VL 分析结果"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM vl_analysis WHERE image_id = ?", (image_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('objects'):
                    result['objects'] = json.loads(result['objects'])
                return result
            return None

    # ==================== 搜索功能 ====================

    def search_images(self, query: str, limit: int = 50) -> List[Dict]:
        """全文搜索图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT i.*, v.ocr_text, v.category, v.scene_description
                FROM images i
                LEFT JOIN vl_analysis v ON i.id = v.image_id
                LEFT JOIN faces f ON i.id = f.image_id
                WHERE i.file_path LIKE ?
                   OR v.ocr_text LIKE ?
                   OR v.category LIKE ?
                   OR v.scene_description LIKE ?
                   OR f.person_name LIKE ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', f'%{query}%',
                  f'%{query}%', f'%{query}%', limit))
            return [dict(row) for row in cursor.fetchall()]

    def search_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """按分类搜索图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.*, v.category
                FROM images i
                JOIN vl_analysis v ON i.id = v.image_id
                WHERE v.category LIKE ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (f'%{category}%', limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """按日期范围搜索图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM images
                WHERE captured_at BETWEEN ? AND ?
                ORDER BY captured_at DESC
            """, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]

    def get_images_grouped_by_date(self, limit: int = 100) -> List[Dict]:
        """按日期分组获取图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT date(captured_at) as date_group,
                       COUNT(*) as count,
                       GROUP_CONCAT(id) as ids
                FROM images
                WHERE processed = TRUE
                GROUP BY date(captured_at)
                ORDER BY date_group DESC
                LIMIT ?
            """, (limit // 10,))
            return [dict(row) for row in cursor.fetchall()]
