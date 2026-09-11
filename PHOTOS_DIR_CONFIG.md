# 照片目录配置说明

## 概述

AutoAlbum 默认使用 `/mnt/nfs/photos` 作为照片存储目录，但系统已配置为在该目录不可访问时自动回退到本地目录，确保 Web 服务不会中断。

## 配置逻辑

### 1. 默认配置

```python
PHOTOS_DIR = Path("/mnt/nfs/photos")  # 默认 NFS 路径
```

### 2. 环境变量覆盖

可以通过环境变量设置自定义照片目录：

```bash
export PHOTOS_DIR=/path/to/your/photos
python -m photo_web.main
```

### 3. 自动回退机制

系统按以下顺序确定照片目录：

1. **环境变量 `PHOTOS_DIR`** - 如果设置，优先使用
2. **默认 NFS 路径 `/mnt/nfs/photos`** - 如果可访问，使用此路径
3. **本地备用目录 `data/photos`** - 如果上述都不可用，使用此路径

### 4. 启动行为

即使 NFS 目录不可访问，Web 服务器也会正常启动，但会显示警告信息：

```
Warning: /mnt/nfs/photos not accessible
Using fallback photos directory: /Users/xzheng4/codebase/autoalbum/data/photos
Starting AutoAlbum Web Server...
  URL: http://localhost:5000
  Database: /Users/xzheng4/codebase/autoalbum/data/autoalbum.db
  Photos: /Users/xzheng4/codebase/autoalbum/data/photos
```

## 使用场景

### 场景 1: NFS 正常挂载

```bash
# NFS 已挂载到 /mnt/nfs/photos
# 系统自动使用 NFS 目录
python -m photo_web.main
```

### 场景 2: NFS 不可用

```bash
# NFS 未挂载或不可访问
# 系统自动回退到 data/photos
python -m photo_web.main
# Web 服务正常启动，但照片文件可能不可用
```

### 场景 3: 使用本地照片目录

```bash
# 使用环境变量指定本地目录
export PHOTOS_DIR=/Volumes/MyPhotos
python -m photo_web.main
```

### 场景 4: Docker 容器部署

```bash
# 挂载外部存储
docker run -v /host/photos:/photos -e PHOTOS_DIR=/photos autoalbum

# 或使用命名卷
docker run -v my_photos:/photos -e PHOTOS_DIR=/photos autoalbum
```

## 数据库说明

数据库 (`autoalbum.db`) 存储图片的**绝对路径**。如果照片目录变更，可能需要：

1. **重新扫描照片目录**:
   ```bash
   python -m photo_analyzer.main scan
   ```

2. **或手动更新数据库中的路径** (高级用户):
   ```sql
   UPDATE images SET file_path = replace(file_path, '/old/path', '/new/path');
   ```

## 故障排除

### 问题 1: Web 服务启动失败

检查日志输出，确认：
- 数据库文件是否存在且可读写
- 端口是否被占用

### 问题 2: 照片无法显示

1. 检查照片目录配置：
   ```bash
   python3 -c "from photo_analyzer.config import PHOTOS_DIR; print(PHOTOS_DIR)"
   ```

2. 验证照片文件是否存在：
   ```bash
   ls -la "$PHOTOS_DIR"
   ```

3. 检查数据库中的路径：
   ```bash
   sqlite3 data/autoalbum.db "SELECT file_path FROM images LIMIT 5;"
   ```

### 问题 3: NFS 挂载问题

检查 NFS 挂载状态：
```bash
mount | grep nfs
df -h /mnt/nfs/photos
```

重新挂载 NFS：
```bash
sudo mount -a
# 或
sudo mount -t nfs nfs_server:/export/photos /mnt/nfs/photos
```

## 配置摘要

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PHOTOS_DIR` | `/mnt/nfs/photos` | 照片存储目录 |
| `DATA_DIR` | `data/` | 数据文件目录 |
| `DATABASE_PATH` | `data/autoalbum.db` | 数据库文件路径 |
| `FACES_DIR` | `data/faces/` | 人脸样本目录 |

## 最佳实践

1. **生产环境**: 使用 NFS 或其他网络存储，确保数据持久性
2. **开发环境**: 使用本地目录，便于调试
3. **备份**: 定期备份 `data/` 目录和数据库
4. **监控**: 监控 NFS 挂载状态，设置告警
