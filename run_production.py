"""
Production server runner using Waitress
"""

from waitress import serve
from app import app

host = "0.0.0.0"
port = 10100

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    waitress_logger = logging.getLogger("waitress")
    waitress_logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting production server on {host}:{port}")

    serve(
        app,
        host=host,
        port=port,
        _quiet=False,
    )
