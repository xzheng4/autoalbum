"""
PhotoWeb 启动入口
"""
import argparse
from .app import run_server


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="AutoAlbum Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind to")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
