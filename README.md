# AutoAlbum - 家庭相册管理系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.16.0-green.svg)](https://github.com/vllm-project/vllm)
[![Flask](https://img.shields.io/badge/Flask-3.0+-red.svg)](https://flask.palletsprojects.com/)

AI 驱动的家庭照片分析与管理系统，支持人脸识别、OCR 提取、图片内容理解、EXIF 信息提取等功能。

## 功能特性

### 照片分析

- **人脸识别** - 基于 insightface buffalo_l 的人脸检测和识别，支持家庭成员注册
- **OCR 文字提取** - 使用 Qwen3-VL-4B 提取图片中的文字
- **图片理解** - 场景描述、分类、物体检测、氛围分析
- **EXIF 信息** - 拍摄设备、日期、GPS 位置等
- **重复检测** - 感知哈希（pHash）算法检测相似照片
- **增量处理** - 自动跳过已处理照片，支持断点续传
- **批量推理** - 支持 vLLM 并行推理，batch_size=1/2/4/8/16/32 可配

### Web 展示

- **照片浏览** - 按日期分组、画廊视图
- **人物相册** - 按人物筛选
- **模糊搜索** - 支持 OCR 文字、分类、人物名搜索
- **照片详情** - 查看完整的分析结果和 EXIF 信息

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 注册家庭成员（可选）

在 `data/faces/` 目录下添加家庭成员照片：

```
data/faces/
├── 张三/
│   └── photo1.jpg
└── 李四/
    └── photo1.jpg
```

运行注册：

```bash
python -m photo_analyzer.main register
```

### 3. 扫描并分析照片

将照片放入 `data/photos/` 目录，然后运行：

```bash
# 扫描并分析所有照片
python -m photo_analyzer.main analyze

# 仅扫描（不分析）
python -m photo_analyzer.main scan

# 查看状态
python -m photo_analyzer.main status
```

### 4. 启动 Web 服务

```bash
python -m photo_web.main
```

访问 http://localhost:5000

## 命令行选项

### 照片分析器

```bash
# 指定 batch size (1/2/4/8/16/32)
python -m photo_analyzer.main analyze --batch-size 4

# 限制处理数量（测试用）
python -m photo_analyzer.main analyze --limit 100

# 指定照片目录
python -m photo_analyzer.main scan --photos-dir /path/to/photos

# 刷新人脸信息（保留 VL 分析）
python -m photo_analyzer.main refresh-faces

# 刷新 VL 分析（保留人脸信息）
python -m photo_analyzer.main refresh-vl
```

### Web 服务器

```bash
# 指定端口
python -m photo_web.main --port 8080

# 调试模式
python -m photo_web.main --debug
```

## 项目结构

```
autoalbum/
├── photo_analyzer/        # 照片分析子项目
│   ├── config.py          # 配置管理
│   ├── database.py        # SQLite 数据库操作
│   ├── models.py          # 数据库模型
│   ├── face_recognition.py # 人脸识别模块
│   ├── exif_extractor.py  # EXIF 提取模块
│   ├── vl_analyzer.py     # Qwen3-VL 分析模块
│   ├── duplicate_detector.py # 重复检测模块
│   ├── scanner.py         # 图片扫描器
│   └── main.py            # 主入口
│
├── photo_web/             # Web 展示子项目
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
├── requirements.txt       # 依赖列表
├── CLAUDE.md             # 详细使用文档
└── README.md
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 人脸识别 | insightface buffalo_l |
| EXIF 提取 | Pillow + exifread |
| OCR/图片分析 | vLLM 0.16.0 + Qwen3-VL-4B-Instruct-FP8 |
| 重复检测 | imagehash (pHash) |
| 数据库 | SQLite3 |
| Web 框架 | Flask |
| 前端 | Bootstrap 5 |

## 系统要求

- GPU: RTX 5060Ti 16GB 或同等
- Python: 3.10+
- 显存：建议 12GB+ 用于批量推理

## 性能优化

- **Batch 推理**: 支持 batch_size=1/2/4/8/16/32，根据显存调整
- **增量处理**: 自动跳过已处理照片
- **图片缩放**: 分析前自动缩放图片到合适尺寸
- **并行处理**: vLLM 批量并行推理，提升吞吐量

## 工具

### 数据库检查工具

```bash
# 查看最后 10 张图片的完整信息
python tools/check_db.py

# 清空人脸数据
python tools/check_db.py --clear-faces

# 清空 VL 分析数据
python tools/check_db.py --clear-vl

# 清空所有数据
python tools/check_db.py --clear-all
```

## 许可证

MIT License
