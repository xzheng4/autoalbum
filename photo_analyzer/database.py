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
            # 先检查是否已存在
            cursor = conn.execute(
                "SELECT id FROM images WHERE file_path = ?", (file_path,)
            )
            row = cursor.fetchone()

            if row:
                # 已存在则更新
                image_id = row[0]
                conn.execute("""
                    UPDATE images SET
                        file_hash = ?,
                        file_size = ?,
                        width = ?,
                        height = ?,
                        format = ?,
                        captured_at = ?
                    WHERE id = ?
                """, (file_hash, file_size, width, height, format, captured_at, image_id))
                return image_id
            else:
                # 不存在则插入
                cursor = conn.execute("""
                    INSERT INTO images
                    (file_path, file_hash, file_size, width, height, format, captured_at, exif_processed, face_processed, vl_processed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, FALSE)
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
        """获取未完成处理的图片（任何阶段未完成）"""
        query = """
            SELECT * FROM images
            WHERE exif_processed = FALSE
               OR face_processed = FALSE
               OR vl_processed = FALSE
            ORDER BY id
        """
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

    def get_random_images(self, limit: int = 10) -> List[Dict]:
        """随机获取指定数量的图片"""
        query = "SELECT * FROM images ORDER BY RANDOM() LIMIT ?"
        with self.get_connection() as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def mark_processed(self, image_id: int):
        """标记图片为已处理（所有阶段）"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE images
                SET exif_processed = TRUE, face_processed = TRUE, vl_processed = TRUE
                WHERE id = ?
            """, (image_id,))

    def mark_exif_processed(self, image_id: int):
        """标记图片 EXIF 处理完成"""
        with self.get_connection() as conn:
            conn.execute("UPDATE images SET exif_processed = TRUE WHERE id = ?", (image_id,))

    def mark_face_processed(self, image_id: int):
        """标记图片人脸处理完成"""
        with self.get_connection() as conn:
            conn.execute("UPDATE images SET face_processed = TRUE WHERE id = ?", (image_id,))

    def mark_vl_processed(self, image_id: int):
        """标记图片 VL 处理完成"""
        with self.get_connection() as conn:
            conn.execute("UPDATE images SET vl_processed = TRUE WHERE id = ?", (image_id,))

    def get_unprocessed_exif_images(self, limit: int = None) -> List[Dict]:
        """获取未完成 EXIF 处理的图片"""
        query = "SELECT * FROM images WHERE exif_processed = FALSE ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_unprocessed_face_images(self, limit: int = None) -> List[Dict]:
        """获取未完成人脸处理的图片"""
        query = "SELECT * FROM images WHERE face_processed = FALSE ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_unprocessed_vl_images(self, limit: int = None) -> List[Dict]:
        """获取未完成 VL 处理的图片"""
        query = "SELECT * FROM images WHERE vl_processed = FALSE ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_random_processed_vl_images(self, limit: int = 10) -> List[Dict]:
        """从已处理 VL 的图片中随机获取指定数量"""
        query = """
            SELECT * FROM images
            WHERE vl_processed = TRUE
            ORDER BY RANDOM()
            LIMIT ?
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def image_exists(self, file_path: str) -> bool:
        """检查图片是否已存在"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM images WHERE file_path = ?", (file_path,)
            )
            return cursor.fetchone() is not None

    def is_processed(self, file_path: str) -> bool:
        """检查图片是否已处理（所有阶段完成）"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT exif_processed, face_processed, vl_processed FROM images
                   WHERE file_path = ?""", (file_path,)
            )
            row = cursor.fetchone()
            return row and all(row)

    def get_image_count(self) -> int:
        """获取图片总数"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM images")
            return cursor.fetchone()[0]

    def get_processed_count(self) -> int:
        """获取已处理图片数量（所有阶段都完成）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM images
                WHERE exif_processed = TRUE
                  AND face_processed = TRUE
                  AND vl_processed = TRUE
            """)
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
                WHERE vl_processed = TRUE
                GROUP BY date(captured_at)
                ORDER BY date_group DESC
                LIMIT ?
            """, (limit // 10,))
            return [dict(row) for row in cursor.fetchall()]

    def get_stage_counts(self) -> Dict[str, int]:
        """获取各阶段处理状态统计"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN exif_processed THEN 1 ELSE 0 END) as exif_done,
                    SUM(CASE WHEN face_processed THEN 1 ELSE 0 END) as face_done,
                    SUM(CASE WHEN vl_processed THEN 1 ELSE 0 END) as vl_done
                FROM images
            """)
            row = cursor.fetchone()
            return {
                "total_images": row["total"] or 0,
                "exif_processed": row["exif_done"] or 0,
                "face_processed": row["face_done"] or 0,
                "vl_processed": row["vl_done"] or 0,
            }

    # ==================== 分类统计功能 ====================

    def get_category_stats(self) -> List[Dict]:
        """获取 VL 分类统计（按 category 字段）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM vl_analysis
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category
                ORDER BY count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_location_stats(self) -> Dict:
        """获取拍摄地点统计（室内/户外，基于 category 字段）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    SUM(CASE WHEN category IN ('室内', 'home', 'indoor') THEN 1 ELSE 0 END) as indoor,
                    SUM(CASE WHEN category IN ('户外', 'outdoor', 'nature', '旅游') THEN 1 ELSE 0 END) as outdoor
                FROM vl_analysis
            """)
            row = cursor.fetchone()
            return {
                "indoor": row["indoor"] or 0,
                "outdoor": row["outdoor"] or 0,
            }

    def get_mood_stats(self) -> List[Dict]:
        """获取氛围统计"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT mood, COUNT(*) as count
                FROM vl_analysis
                WHERE mood IS NOT NULL AND mood != ''
                GROUP BY mood
                ORDER BY count DESC
                LIMIT 10
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_object_stats(self) -> List[Dict]:
        """获取常见物体统计（从 objects JSON 数组中提取）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT objects FROM vl_analysis
                WHERE objects IS NOT NULL
            """)
            rows = cursor.fetchall()

            # 解析 JSON 并统计物体
            from collections import Counter
            import json
            object_counter = Counter()

            for row in rows:
                if row["objects"]:
                    try:
                        objects = json.loads(row["objects"])
                        object_counter.update(objects)
                    except:
                        pass

            return [{"object": obj, "count": cnt} for obj, cnt in object_counter.most_common(20)]

    def get_camera_stats(self) -> List[Dict]:
        """获取拍摄设备统计"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT make, model, COUNT(*) as count
                FROM exif_data
                WHERE make IS NOT NULL
                GROUP BY make, model
                ORDER BY count DESC
                LIMIT 10
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_year_stats(self) -> List[Dict]:
        """获取按年份统计的图片数量（基于 images.captured_at）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    CASE
                        WHEN captured_at IS NULL THEN '未知'
                        WHEN captured_at = '' THEN '未知'
                        WHEN strftime('%Y', captured_at) IS NULL THEN '未知'
                        ELSE strftime('%Y', captured_at)
                    END as year,
                    COUNT(*) as count
                FROM images
                GROUP BY year
                ORDER BY
                    CASE WHEN year = '未知' THEN 1 ELSE 0 END,
                    year DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_year(self, year: str, limit: int = 100) -> List[Dict]:
        """按年份获取图片（基于 images.captured_at）"""
        if year == "未知":
            query = """
                SELECT i.*, v.category
                FROM images i
                LEFT JOIN vl_analysis v ON i.id = v.image_id
                WHERE i.captured_at IS NULL
                   OR i.captured_at = ''
                   OR strftime('%Y', i.captured_at) IS NULL
                ORDER BY i.id DESC
                LIMIT ?
            """
            params = (limit,)
        else:
            query = """
                SELECT i.*, v.category
                FROM images i
                LEFT JOIN vl_analysis v ON i.id = v.image_id
                WHERE strftime('%Y', i.captured_at) = ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """
            params = (year, limit)

        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_person_stats_with_images(self) -> List[Dict]:
        """获取人物统计及预览图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT f.person_name, COUNT(DISTINCT f.image_id) as count
                FROM faces f
                GROUP BY f.person_name
                ORDER BY count DESC
            """)
            persons = [dict(row) for row in cursor.fetchall()]

        # 为每个人物获取一张预览图
        for person in persons:
            images = self.get_images_by_person(person["person_name"])
            person["preview_image"] = images[0] if images else None
            person["count"] = len(images)

        return persons

    def get_camera_stats_detailed(self) -> List[Dict]:
        """获取拍摄设备统计（带预览图）"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT e.make, e.model, COUNT(DISTINCT e.image_id) as count,
                       (SELECT image_id FROM exif_data e2
                        WHERE e2.make = e.make AND e2.model = e.model
                        LIMIT 1) as sample_image_id
                FROM exif_data e
                WHERE e.make IS NOT NULL
                GROUP BY e.make, e.model
                ORDER BY count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """按分类获取图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.*, v.category, v.scene_description
                FROM images i
                JOIN vl_analysis v ON i.id = v.image_id
                WHERE v.category = ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (category, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_mood(self, mood: str, limit: int = 100) -> List[Dict]:
        """按氛围获取图片"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.*, v.mood
                FROM images i
                JOIN vl_analysis v ON i.id = v.image_id
                WHERE v.mood = ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (mood, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_images_by_location_type(self, location_type: str, limit: int = 100) -> List[Dict]:
        """按地点类型（室内/户外）获取图片"""
        if location_type == "indoor":
            categories = "('室内', 'home', 'indoor')"
        else:
            categories = "('户外', 'outdoor', 'nature', '旅游')"

        with self.get_connection() as conn:
            cursor = conn.execute(f"""
                SELECT i.*, v.category
                FROM images i
                JOIN vl_analysis v ON i.id = v.image_id
                WHERE v.category IN {categories}
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
