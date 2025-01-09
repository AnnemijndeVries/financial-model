import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_file="model_training.log", level=logging.INFO):
    logger = logging.getLogger("ModelTrainingLogger")

    # Controleer of er al handlers zijn
    if not logger.handlers:
        # Formatter instellen
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

        # File handler
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Log level instellen
        logger.setLevel(level)

    return logger
