"""
数据库模型定义
"""

DB_SCHEMA = """
-- 图片主表
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    format TEXT,
    captured_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

-- EXIF 信息
CREATE TABLE IF NOT EXISTS exif_data (
    image_id INTEGER,
    make TEXT,
    model TEXT,
    lens_model TEXT,
    iso INTEGER,
    aperture REAL,
    shutter_speed TEXT,
    focal_length REAL,
    gps_lat REAL,
    gps_lon REAL,
    gps_alt REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- 人脸信息
CREATE TABLE IF NOT EXISTS faces (
    id PRIMARY KEY,
    image_id INTEGER,
    person_name TEXT,
    face_encoding BLOB,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    confidence REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- OCR 文本和图片理解结果
CREATE TABLE IF NOT EXISTS vl_analysis (
    image_id INTEGER,
    ocr_text TEXT,
    scene_description TEXT,
    category TEXT,
    objects TEXT,  -- JSON 数组
    mood TEXT,
    confidence REAL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- 重复照片检测（感知哈希）
CREATE TABLE IF NOT EXISTS duplicates (
    id INTEGER PRIMARY KEY,
    image_id_1 INTEGER,
    image_id_2 INTEGER,
    similarity REAL,
    hash_1 TEXT,
    hash_2 TEXT,
    FOREIGN KEY (image_id_1) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id_2) REFERENCES images(id) ON DELETE CASCADE
);

-- 全文搜索索引 (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    file_path,
    ocr_text,
    scene_description,
    category,
    person_name,
    content='images',
    content_rowid='id'
);

-- 触发器：自动更新全文索引
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, file_path) VALUES (new.id, new.file_path);
END;

CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, file_path) VALUES('delete', old.id, old.file_path);
END;

CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, file_path) VALUES('delete', old.id, old.file_path);
    INSERT INTO images_fts(rowid, file_path) VALUES (new.id, new.file_path);
END;

-- 索引优化
CREATE INDEX IF NOT EXISTS idx_images_processed ON images(processed);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(file_path);
CREATE INDEX IF NOT EXISTS idx_images_captured_at ON images(captured_at);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_name);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_vl_image ON vl_analysis(image_id);
CREATE INDEX IF NOT EXISTS idx_vl_category ON vl_analysis(category);
CREATE INDEX IF NOT EXISTS idx_vl_ocr ON vl_analysis(ocr_text);
CREATE INDEX IF NOT EXISTS idx_exif_image ON exif_data(image_id);
"""
