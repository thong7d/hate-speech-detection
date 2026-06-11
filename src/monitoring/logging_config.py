import os
import logging
from logging.handlers import RotatingFileHandler

def configure_logging(level: int = logging.INFO, log_file: str = "logs/app.log") -> None:
    """
    Configures unified logging system with stdout and rotating file handlers.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clean existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%dT%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Rotate logs if size exceeds 5MB, keeping 3 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logging.info(f"Logging configured at level={logging.getLevelName(level)}. Logs written to console and {log_file}")
