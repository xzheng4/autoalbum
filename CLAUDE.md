# CLAUDE.md

AutoAlbum - 家庭相册管理系统

## 项目概述

AutoAlbum 是一个家庭照片管理系统，包含两个子项目：

1. **photo_analyzer** - 照片分析与入库
2. **photo_web** - Web 展示系统

## 环境要求

- GPU: RTX 5060Ti 16GB
- Python 3.10+
- vLLM 0.16.0
- Qwen3-VL-4B 模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 照片分析器 (photo_analyzer)

### 注册家庭成员人脸

在 `data/faces/` 目录下按人名创建子目录，放入该成员的照片：

```
data/faces/
├── 张三/
│   ├── photo1.jpg
│   └── photo2.jpg
└── 李四/
    └── photo1.jpg
```

然后运行：

```bash
python -m photo_analyzer.main register
```

### 扫描照片

```bash
# 扫描 photos 目录，添加新图片到数据库
python -m photo_analyzer.main scan
```

### 分析照片

```bash
# 分析所有未处理的照片（支持增量处理）
python -m photo_analyzer.main analyze

# 指定 batch size (1/2/4/8/16/32)
python -m photo_analyzer.main analyze --batch-size 4

# 限制处理数量（用于测试）
python -m photo_analyzer.main analyze --limit 100
```

### 查看状态

```bash
python -m photo_analyzer.main status
```

### 配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | vLLM 批处理大小 | 4 |
| `--limit` | 限制处理图片数 | 无限制 |
| `--db` | 数据库路径 | data/autoalbum.db |
| `--photos-dir` | 照片目录 | data/photos |

## Web 展示系统 (photo_web)

### 启动服务器

```bash
# 默认端口 5000
python -m photo_web.main

# 指定端口
python -m photo_web.main --port 8080

# 调试模式
python -m photo_web.main --debug
```

访问：http://localhost:5000

### 功能

- **首页** - 按日期分组展示照片
- **画廊** - 缩略图浏览所有照片
- **人物** - 按人物查看照片
- **搜索** - 模糊搜索（OCR、分类、人物名）

## 项目结构

```
autoalbum/
├── photo_analyzer/        # 照片分析子项目
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── database.py        # SQLite 数据库操作
│   ├── models.py          # 数据库模型
│   ├── face_recognition.py # 人脸识别
│   ├── exif_extractor.py  # EXIF 提取
│   ├── vl_analyzer.py     # Qwen3-VL 分析
│   ├── duplicate_detector.py # 重复检测
│   ├── scanner.py         # 图片扫描
│   └── main.py            # 主入口
│
├── photo_web/             # Web 展示子项目
│   ├── __init__.py
│   ├── app.py             # Flask 应用
│   ├── main.py            # 启动入口
│   ├── templates/         # HTML 模板
│   └── static/            # 静态资源
│
├── data/
│   ├── photos/            # 原始照片目录
│   ├── faces/             # 家庭成员人脸样本
│   └── autoalbum.db       # SQLite 数据库
│
└── requirements.txt       # 依赖
```

## 数据库 Schema

- **images** - 图片主表
- **exif_data** - EXIF 信息
- **faces** - 人脸信息
- **vl_analysis** - VL 分析结果（OCR、分类等）
- **duplicates** - 重复照片记录

## 核心功能

### 照片分析
1. **EXIF 提取** - 拍摄设备、日期、GPS 等
2. **人脸识别** - 家庭成员识别与标注
3. **OCR 分析** - 图片文字提取（Qwen3-VL）
4. **图片理解** - 场景描述、分类、物体检测（Qwen3-VL）
5. **重复检测** - 感知哈希（pHash）去重

### 增量处理
- 自动跳过已处理照片
- 支持断点续传
- 通过文件路径和哈希值判断

## 开发说明

### 添加新的分析模块

在 `photo_analyzer/` 下创建新模块，然后在 `main.py` 中集成。

### 修改数据库 Schema

编辑 `models.py` 中的 `DB_SCHEMA`，数据库会自动应用变更。

### Web 模板

使用 Bootstrap 5 和 Jinja2 模板引擎。
