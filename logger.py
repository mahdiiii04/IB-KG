import logging
import os

class Logger:
    def __init__(self, log_dir="logs", log_filename="train.log", level=logging.INFO):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        # Configure logger
        self.logger = logging.getLogger("ModelLogger")
        self.logger.setLevel(level)
        self.logger.handlers = []  # Prevent duplicate handlers if re-run

        # Formatter
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def progress_bar(self, current, total, bar_length=30):
        percent = int(100 * (current / total))
        bar = '|' * int(bar_length * (current / total))
        spaces = ' ' * (bar_length - len(bar))
        self.info(f"[{bar}{spaces}] {percent}% ({current}/{total})")
