#!/usr/bin/env python3
"""
数据库检查工具 - 从 autoalbum.db 中抽取最后 10 张图片的所有信息
用于开发者检查数据库内容和分析结果
"""
import sys
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from photo_analyzer.config import DATABASE_PATH
from photo_analyzer.database import Database


def print_separator(title: str = ""):
    """打印分隔线"""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_image_info(image: dict, vl_analysis: dict, faces: list, exif: dict):
    """打印单张图片的完整信息"""

    # 基本信息
    print(f"\n【图片 ID: {image['id']}】")
    print(f"文件路径：{image['file_path']}")
    print(f"文件名：{Path(image['file_path']).name}")

    if image.get('file_size'):
        size_mb = image['file_size'] / 1024 / 1024
        print(f"文件大小：{size_mb:.2f} MB")

    if image.get('width') and image.get('height'):
        print(f"图片尺寸：{image['width']} x {image['height']}")

    if image.get('format'):
        print(f"格式：{image['format']}")

    if image.get('captured_at'):
        print(f"拍摄时间：{image['captured_at']}")

    if image.get('file_hash'):
        print(f"文件哈希：{image['file_hash'][:16]}...")

    print(f"处理状态：{'已处理' if image.get('processed') else '未处理'}")

    # VL 分析结果
    if vl_analysis:
        print_separator("VL 分析结果")

        if vl_analysis.get('category'):
            print(f"类别：{vl_analysis['category']}")

        if vl_analysis.get('scene_description'):
            print(f"场景描述：{vl_analysis['scene_description'][:200]}")

        if vl_analysis.get('ocr_text'):
            print(f"OCR 文字：{vl_analysis['ocr_text'][:200]}")

        if vl_analysis.get('objects'):
            objects = vl_analysis.get('objects', [])
            print(f"检测到物体：{', '.join(objects[:10])}{'...' if len(objects) > 10 else ''}")

        if vl_analysis.get('mood'):
            print(f"氛围：{vl_analysis['mood']}")

        if vl_analysis.get('confidence'):
            print(f"置信度：{vl_analysis['confidence']:.2%}")
    else:
        print("\n[暂无 VL 分析结果]")

    # 人脸信息
    if faces:
        print_separator("人脸信息")
        for i, face in enumerate(faces, 1):
            person = face.get('person_name', '未知')
            confidence = face.get('confidence', 0)
            bbox = (face.get('bbox_x'), face.get('bbox_y'),
                    face.get('bbox_w'), face.get('bbox_h'))
            print(f"  人脸 {i}: {person} (置信度：{confidence:.2%}) 位置：{bbox}")
    else:
        print("\n[未检测到人脸]")

    # EXIF 信息
    if exif and any(exif.values()):
        print_separator("EXIF 信息")

        if exif.get('make'):
            print(f"相机品牌：{exif['make']}")

        if exif.get('model'):
            print(f"相机型号：{exif['model']}")

        if exif.get('lens_model'):
            print(f"镜头型号：{exif['lens_model']}")

        if exif.get('iso'):
            print(f"ISO: {exif['iso']}")

        if exif.get('aperture'):
            print(f"光圈：f/{exif['aperture']}")

        if exif.get('shutter_speed'):
            print(f"快门：{exif['shutter_speed']}")

        if exif.get('focal_length'):
            print(f"焦距：{exif['focal_length']}mm")

        if exif.get('gps_lat') and exif.get('gps_lon'):
            print(f"GPS 坐标：{exif['gps_lat']:.4f}, {exif['gps_lon']:.4f}")
    else:
        print("\n[暂无 EXIF 信息]")


def check_database(db_path: str = None, limit: int = 10):
    """
    检查数据库，抽取最后处理的照片信息

    Args:
        db_path: 数据库路径
        limit: 抽取的图片数量
    """
    db_path = db_path or str(DATABASE_PATH)

    print_separator("AutoAlbum 数据库检查工具")
    print(f"数据库路径：{db_path}")

    try:
        db = Database(db_path)
    except Exception as e:
        print(f"错误：无法打开数据库 - {e}")
        return

    # 获取统计信息
    total_count = db.get_image_count()
    processed_count = db.get_processed_count()
    persons = db.get_all_persons()

    print(f"\n数据库统计:")
    print(f"  总图片数：{total_count}")
    print(f"  已处理：{processed_count}")
    print(f"  未处理：{total_count - processed_count}")
    print(f"  家庭成员：{len(persons)} 人 ({', '.join(persons) if persons else '无'})")

    # 获取最后 N 张图片
    images = db.get_all_images(limit=limit)

    if not images:
        print("\n数据库中没有图片记录")
        return

    print(f"\n正在显示最后 {len(images)} 张图片的详细信息...")

    for image in images:
        image_id = image['id']

        # 获取关联数据
        vl_analysis = db.get_vl_analysis(image_id)
        faces = db.get_faces_by_image(image_id)
        exif = db.get_exif_data(image_id)

        # 打印信息
        print_image_info(image, vl_analysis, faces, exif)

    # 打印总结
    print_separator("检查完成")
    print(f"已显示数据库中最后 {len(images)} 张图片的完整信息")
    print(f"总共有 {total_count} 张图片")


def clear_faces(db_path: str = None, confirm: bool = False):
    """
    清空所有人脸数据

    Args:
        db_path: 数据库路径
        confirm: 是否需要确认
    """
    db_path = db_path or str(DATABASE_PATH)

    print_separator("清空人脸数据")
    print(f"数据库路径：{db_path}")

    if not confirm:
        response = input("\n确定要清空所有人脸数据吗？此操作不可恢复！(y/N): ")
        if response.lower() != 'y':
            print("已取消操作")
            return

    try:
        db = Database(db_path)
        with db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM faces")
            deleted = cursor.rowcount
            conn.commit()

        print(f"\n已删除 {deleted} 条人脸记录")
        print("人脸数据已清空，可以重新运行分析来识别人脸")
    except Exception as e:
        print(f"错误：{e}")


def clear_vl_analysis(db_path: str = None, confirm: bool = False):
    """
    清空所有 VL 分析数据（OCR、图片分类等）

    Args:
        db_path: 数据库路径
        confirm: 是否需要确认
    """
    db_path = db_path or str(DATABASE_PATH)

    print_separator("清空 VL 分析数据")
    print(f"数据库路径：{db_path}")

    if not confirm:
        response = input("\n确定要清空所有 VL 分析数据吗？此操作不可恢复！(y/N): ")
        if response.lower() != 'y':
            print("已取消操作")
            return

    try:
        db = Database(db_path)
        with db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM vl_analysis")
            deleted = cursor.rowcount
            conn.commit()

        print(f"\n已删除 {deleted} 条 VL 分析记录")
        print("VL 分析数据已清空，可以重新运行分析来进行 OCR 和图片理解")
    except Exception as e:
        print(f"错误：{e}")


def clear_all_data(db_path: str = None, confirm: bool = False):
    """
    清空所有数据（恢复到初始状态）

    Args:
        db_path: 数据库路径
        confirm: 是否需要确认
    """
    db_path = db_path or str(DATABASE_PATH)

    print_separator("清空所有数据")
    print(f"数据库路径：{db_path}")
    print("\n警告：这将删除所有图片记录、人脸数据、VL 分析结果和 EXIF 信息！")

    if not confirm:
        response = input("\n确定要清空所有数据吗？此操作不可恢复！(y/N): ")
        if response.lower() != 'y':
            print("已取消操作")
            return

    try:
        db = Database(db_path)
        with db.get_connection() as conn:
            # 按顺序删除（注意外键约束）
            conn.execute("DELETE FROM vl_analysis")
            conn.execute("DELETE FROM faces")
            conn.execute("DELETE FROM exif_data")
            conn.execute("DELETE FROM duplicates")
            conn.execute("DELETE FROM images")
            conn.commit()

        print("\n所有数据已清空")
        print("请重新运行扫描和分析命令：")
        print("  python -m photo_analyzer.main scan")
        print("  python -m photo_analyzer.main analyze")
    except Exception as e:
        print(f"错误：{e}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="检查 AutoAlbum 数据库")
    parser.add_argument("--db", type=str, help="数据库路径")
    parser.add_argument("--limit", type=int, default=10,
                        help="要检查的图片数量（默认 10）")

    # 清空数据选项
    clear_group = parser.add_argument_group("清空数据选项")
    clear_group.add_argument("--clear-all", action="store_true",
                             help="清空所有数据（图片记录、人脸、VL 分析、EXIF）")
    clear_group.add_argument("--clear-faces", action="store_true",
                             help="清空所有人脸数据")
    clear_group.add_argument("--clear-vl", action="store_true",
                             help="清空所有 VL 分析数据（OCR、分类等）")
    clear_group.add_argument("-y", "--yes", action="store_true",
                             help="确认操作，不询问")

    args = parser.parse_args()

    # 处理清空操作
    if args.clear_all:
        clear_all_data(db_path=args.db, confirm=args.yes)
    elif args.clear_faces:
        clear_faces(db_path=args.db, confirm=args.yes)
    elif args.clear_vl:
        clear_vl_analysis(db_path=args.db, confirm=args.yes)
    else:
        # 默认检查数据库
        check_database(db_path=args.db, limit=args.limit)


if __name__ == "__main__":
    main()
