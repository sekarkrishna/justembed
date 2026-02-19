"""
CLI — justembed begin
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="justembed", description="JustEmbed — LENS")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    begin_parser = subparsers.add_parser("begin", help="Start JustEmbed server")
    begin_parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Path to working folder",
    )
    begin_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5424,
        help="Server port (default: 5424)",
    )
    begin_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if args.cmd == "begin":
        import uvicorn
        from justembed.config import set_workspace
        from justembed.app import create_app

        if args.workspace:
            set_workspace(args.workspace)
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
