"""
家庭相册 Web 展示应用
"""
import os
import base64
import io
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for
from PIL import Image

from photo_analyzer.config import DATABASE_PATH, WEB_CONFIG, PHOTOS_DIR
from photo_analyzer.database import Database


app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# 配置
app.config["THUMBNAILS_PER_PAGE"] = WEB_CONFIG.get("thumbnails_per_page", 48)
app.config["MAX_THUMBNAIL_SIZE"] = (400, 400)

# 初始化数据库
db = Database(str(DATABASE_PATH))


def get_thumbnail(image_path: str, size: tuple = None) -> bytes:
    """生成缩略图"""
    if size is None:
        size = app.config["MAX_THUMBNAIL_SIZE"]

    try:
        with Image.open(image_path) as img:
            # 转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 生成缩略图
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # 保存到内存
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return buffered.getvalue()
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return None


def image_to_base64(image_path: str, size: tuple = None) -> str:
    """将图片转换为 base64"""
    thumbnail = get_thumbnail(image_path, size)
    if thumbnail:
        return base64.b64encode(thumbnail).decode('utf-8')
    return ""


@app.route("/")
def index():
    """首页 - 按日期分组展示"""
    # 获取统计信息
    total_images = db.get_image_count()
    processed_images = db.get_processed_count()
    persons = db.get_all_persons()

    # 获取按日期分组的图片
    date_groups = db.get_images_grouped_by_date(limit=20)

    # 为每个日期组获取预览图片
    for group in date_groups:
        ids = group.get('ids', '')
        if ids:
            # 获取该日期的第一张图片作为预览
            preview_ids = ids.split(',')[:3]
            group['preview_images'] = []
            for img_id in preview_ids:
                img = db.get_image_by_id(int(img_id))
                if img and os.path.exists(img['file_path']):
                    group['preview_images'].append({
                        'id': img['id'],
                        'thumbnail': image_to_base64(img['file_path'])
                    })

    return render_template("index.html",
                           total_images=total_images,
                           processed_images=processed_images,
                           persons=persons,
                           date_groups=date_groups)


@app.route("/gallery")
def gallery():
    """画廊视图 - 缩略图浏览"""
    page = request.args.get('page', 1, type=int)
    limit = app.config["THUMBNAILS_PER_PAGE"]
    offset = (page - 1) * limit

    images = db.get_all_images(limit=limit, offset=offset)
    total = db.get_image_count()
    total_pages = (total + limit - 1) // limit

    # 生成缩略图
    for img in images:
        if os.path.exists(img['file_path']):
            img['thumbnail'] = image_to_base64(img['file_path'])

    return render_template("gallery.html",
                           images=images,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route("/search")
def search():
    """搜索页面"""
    query = request.args.get('q', '')
    results = []

    if query:
        results = db.search_images(query, limit=100)
        for img in results:
            if os.path.exists(img['file_path']):
                img['thumbnail'] = image_to_base64(img['file_path'])

    return render_template("search.html",
                           query=query,
                           results=results,
                           total=len(results))


@app.route("/person/<name>")
def person(name):
    """按人物查看"""
    images = db.get_images_by_person(name)
    for img in images:
        if os.path.exists(img['file_path']):
            img['thumbnail'] = image_to_base64(img['file_path'])

    return render_template("person.html",
                           person_name=name,
                           images=images,
                           total=len(images))


@app.route("/persons")
def persons():
    """所有人物列表"""
    persons_list = db.get_all_persons()
    persons_data = []

    for name in persons_list:
        images = db.get_images_by_person(name)
        count = len(images)

        # 获取一张预览图
        preview = None
        if images and os.path.exists(images[0]['file_path']):
            preview = image_to_base64(images[0]['file_path'])

        persons_data.append({
            'name': name,
            'count': count,
            'preview': preview
        })

    return render_template("persons.html", persons=persons_data)


@app.route("/image/<int:image_id>")
def image_detail(image_id):
    """单张图片详情"""
    img = db.get_image_by_id(image_id)
    if not img:
        return "Image not found", 404

    # 获取 VL 分析结果
    vl_result = db.get_vl_analysis(image_id)

    # 获取人脸信息
    faces = db.get_faces_by_image(image_id)

    # 获取 EXIF 数据
    exif = db.get_exif_data(image_id)

    # 加载原图信息
    file_exists = os.path.exists(img['file_path'])
    full_image = None
    if file_exists:
        full_image = image_to_base64(img['file_path'], size=(1200, 1200))

    return render_template("image_detail.html",
                           image=img,
                           vl_analysis=vl_result,
                           faces=faces,
                           exif=exif,
                           file_exists=file_exists,
                           full_image=full_image)


@app.route("/image/<int:image_id>/full")
def image_full(image_id):
    """下载/查看原图"""
    img = db.get_image_by_id(image_id)
    if not img or not os.path.exists(img['file_path']):
        return "Image not found", 404

    return send_file(img['file_path'])


@app.route("/category/<category>")
def category(category):
    """按分类查看"""
    images = db.search_by_category(category, limit=100)
    for img in images:
        if os.path.exists(img['file_path']):
            img['thumbnail'] = image_to_base64(img['file_path'])

    return render_template("category.html",
                           category=category,
                           images=images,
                           total=len(images))


@app.route("/api/stats")
def api_stats():
    """API - 获取统计信息"""
    return jsonify({
        "total_images": db.get_image_count(),
        "processed_images": db.get_processed_count(),
        "persons": len(db.get_all_persons()),
    })


@app.route("/browse")
def browse():
    """浏览页面 - 左边分类导航，右边图片列表"""
    # 获取所有分类统计
    category_stats = db.get_category_stats()
    location_stats = db.get_location_stats()
    mood_stats = db.get_mood_stats()
    object_stats = db.get_object_stats()
    camera_stats = db.get_camera_stats()
    year_stats = db.get_year_stats()
    persons = db.get_all_persons()

    # 获取人物图片数量
    persons_data = []
    for name in persons:
        count = len(db.get_images_by_person(name))
        persons_data.append({"name": name, "count": count})

    return render_template("browse.html",
                           category_stats=category_stats,
                           location_stats=location_stats,
                           mood_stats=mood_stats,
                           object_stats=object_stats,
                           camera_stats=camera_stats,
                           year_stats=year_stats,
                           persons=persons_data)


@app.route("/images")
def images_list():
    """图片列表 API - 支持各种筛选"""
    # 获取筛选参数
    category = request.args.get("category", "")
    mood = request.args.get("mood", "")
    location = request.args.get("location", "")
    person = request.args.get("person", "")
    year = request.args.get("year", "")
    obj = request.args.get("object", "")
    limit = request.args.get("limit", 100, type=int)

    images = []

    if category:
        images = db.get_images_by_category(category, limit)
    elif mood:
        images = db.get_images_by_mood(mood, limit)
    elif location:
        images = db.get_images_by_location_type(location, limit)
    elif person:
        images = db.get_images_by_person(person)
    elif year:
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT i.*, v.category
                FROM images i
                JOIN vl_analysis v ON i.id = v.image_id
                WHERE strftime('%Y', i.captured_at) = ?
                ORDER BY i.captured_at DESC
                LIMIT ?
            """, (year, limit))
            images = [dict(row) for row in cursor.fetchall()]
    else:
        images = db.get_all_images(limit=limit)

    # 生成缩略图
    for img in images:
        if os.path.exists(img['file_path']):
            img['thumbnail'] = image_to_base64(img['file_path'])

    return render_template("images.html", images=images, total=len(images))


@app.route("/api/categories")
def api_categories():
    """API - 获取所有分类统计"""
    return jsonify({
        "categories": db.get_category_stats(),
        "locations": db.get_location_stats(),
        "moods": db.get_mood_stats(),
        "objects": db.get_object_stats()[:10],
        "cameras": db.get_camera_stats(),
        "years": db.get_year_stats(),
        "persons": [{"name": n, "count": len(db.get_images_by_person(n))}
                    for n in db.get_all_persons()],
    })


@app.route("/api/search")
def api_search():
    """API - 搜索"""
    query = request.args.get('q', '')
    results = db.search_images(query, limit=50)
    return jsonify([{
        "id": r['id'],
        "file_path": r['file_path'],
        "captured_at": r['captured_at'],
        "ocr_text": r.get('ocr_text', ''),
        "category": r.get('category', ''),
    } for r in results])


# 错误页面
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="页面未找到"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="服务器错误"), 500


def run_server(host=None, port=None, debug=None):
    """运行 Web 服务器"""
    host = host or WEB_CONFIG.get("host", "0.0.0.0")
    port = port or WEB_CONFIG.get("port", 5000)
    debug = debug if debug is not None else WEB_CONFIG.get("debug", False)

    print(f"Starting AutoAlbum Web Server...")
    print(f"  URL: http://localhost:{port}")
    print(f"  Photos: {PHOTOS_DIR}")
    print(f"  Database: {DATABASE_PATH}")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
