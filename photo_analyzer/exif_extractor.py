"""
EXIF 信息提取模块
"""
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
import json


class EXIFExtractor:
    """EXIF 信息提取器"""

    def __init__(self):
        pass

    def extract_exif(self, image_path: str) -> Dict[str, Any]:
        """
        提取图片的 EXIF 信息

        Returns:
            dict: 包含 EXIF 信息的字典
        """
        result = {
            'make': None,
            'model': None,
            'lens_model': None,
            'iso': None,
            'aperture': None,
            'shutter_speed': None,
            'focal_length': None,
            'gps_lat': None,
            'gps_lon': None,
            'gps_alt': None,
            'width': None,
            'height': None,
            'format': None,
            'captured_at': None,
        }

        try:
            # 使用 Pillow 获取基本信息和部分 EXIF
            with Image.open(image_path) as img:
                # 基本信息
                result['width'] = img.width
                result['height'] = img.height
                result['format'] = img.format

                # 提取 EXIF 数据
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == 'Make':
                            result['make'] = value
                        elif tag == 'Model':
                            result['model'] = value
                        elif tag == 'ISOSpeedRatings':
                            result['iso'] = value
                        elif tag == 'FNumber':
                            result['aperture'] = self._convert_ifd_rational(value)
                        elif tag == 'ExposureTime':
                            result['shutter_speed'] = str(value)
                        elif tag == 'FocalLength':
                            result['focal_length'] = self._convert_ifd_rational(value)
                        elif tag == 'LensModel':
                            result['lens_model'] = value
                        elif tag == 'DateTimeOriginal':
                            # 格式：'YYYY:MM:DD HH:MM:SS' -> 'YYYY-MM-DD HH:MM:SS'
                            if value:
                                result['captured_at'] = self._format_datetime(value)
                        elif tag == 'GPSInfo':
                            gps_data = self._parse_gps_info(value)
                            result.update(gps_data)

            # 使用 exifread 获取更详细的信息（如果 Pillow 未能完整提取）
            if not result['captured_at']:
                try:
                    with open(image_path, 'rb') as f:
                        tags = exifread.process_file(f, stop_tag='DateTimeOriginal')
                        if 'EXIF DateTimeOriginal' in tags:
                            result['captured_at'] = self._format_datetime(str(tags['EXIF DateTimeOriginal']))
                except Exception:
                    pass

        except Exception as e:
            print(f"Error extracting EXIF from {image_path}: {e}")

        return result

    def _parse_gps_info(self, gps_info: dict) -> Dict[str, float]:
        """解析 GPS 信息"""
        result = {'gps_lat': None, 'gps_lon': None, 'gps_alt': None}

        try:
            if 2 in gps_info and 3 in gps_info:
                # 纬度
                lat_ref = gps_info.get(1, 'N')
                lat_vals = gps_info[2]
                lat = self._convert_to_degrees(lat_vals)
                if lat_ref != 'N':
                    lat = -lat
                result['gps_lat'] = lat

                # 经度
                lon_ref = gps_info.get(3, 'E')
                lon_vals = gps_info[4]
                lon = self._convert_to_degrees(lon_vals)
                if lon_ref != 'E':
                    lon = -lon
                result['gps_lon'] = lon

            if 6 in gps_info:
                # 海拔
                alt = gps_info[6]
                if isinstance(alt, tuple) and len(alt) == 2:
                    result['gps_alt'] = alt[0] / alt[1]
                else:
                    result['gps_alt'] = float(alt) if alt else None

        except Exception as e:
            print(f"Error parsing GPS info: {e}")

        return result

    def _convert_ifd_rational(self, value) -> Optional[float]:
        """
        转换 IFDRational 对象为 float

        IFDRational 是 PIL 处理 EXIF 有理数的类型，可能是：
        - IFDRational 对象（有 num/denom 属性）
        - tuple (numerator, denominator)
        - 普通数字
        """
        if value is None:
            return None

        # 如果是普通数字
        if isinstance(value, (int, float)):
            return float(value)

        # 如果是 tuple
        if isinstance(value, tuple) and len(value) == 2:
            try:
                return float(value[0]) / float(value[1])
            except (ZeroDivisionError, TypeError):
                return None

        # 如果是 IFDRational 对象（有 num 和 denom 属性）
        if hasattr(value, 'num') and hasattr(value, 'denom'):
            try:
                return float(value.num) / float(value.denom)
            except (ZeroDivisionError, TypeError):
                return None

        # 如果是 IFDRational 对象（有 numerator 和 denominator 属性）
        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            try:
                return float(value.numerator) / float(value.denominator)
            except (ZeroDivisionError, TypeError):
                return None

        # 其他情况尝试直接转换
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _format_datetime(self, value: str) -> Optional[str]:
        """
        格式化 EXIF 日期时间字符串

        输入格式：'YYYY:MM:DD HH:MM:SS' 或 'YYYY-MM-DD HH:MM:SS'
        输出格式：'YYYY-MM-DD HH:MM:SS'
        """
        if not value:
            return None

        try:
            # 替换日期部分的冒号为短横线
            # 格式：'2023:05:15 14:30:45' -> '2023-05-15 14:30:45'
            parts = value.split(' ')
            if len(parts) == 2:
                date_part = parts[0].replace(':', '-')
                time_part = parts[1]  # 时间部分保持不变
                return f"{date_part} {time_part}"
            else:
                # 如果没有空格分隔，尝试直接替换前两个冒号
                return value.replace(':', '-', 2)
        except Exception:
            return value

    def _convert_to_degrees(self, value: tuple) -> float:
        """将 GPS 坐标转换为十进制度数"""
        try:
            def convert_rational(r):
                """转换单个 IFDRational 值"""
                if isinstance(r, (int, float)):
                    return float(r)
                if isinstance(r, tuple) and len(r) == 2:
                    return float(r[0]) / float(r[1]) if r[1] != 0 else 0
                if hasattr(r, 'num') and hasattr(r, 'denom'):
                    return float(r.num) / float(r.denom) if r.denom != 0 else 0
                if hasattr(r, 'numerator') and hasattr(r, 'denominator'):
                    return float(r.numerator) / float(r.denominator) if r.denominator != 0 else 0
                return float(r)

            d = convert_rational(value[0]) if len(value) > 0 else 0
            m = convert_rational(value[1]) if len(value) > 1 else 0
            s = convert_rational(value[2]) if len(value) > 2 else 0

            return d + (m / 60.0) + (s / 3600.0)
        except (ZeroDivisionError, IndexError, TypeError):
            return 0.0

    def get_file_hash(self, image_path: str) -> str:
        """计算文件哈希（用于去重和增量检测）"""
        import hashlib

        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_file_size(self, image_path: str) -> int:
        """获取文件大小"""
        return Path(image_path).stat().st_size
