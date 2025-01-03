import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_file="model_training.log", level=logging.INFO):
    # Formatter voor logberichten
    formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Logger instellen
    logger = logging.getLogger("ModelTrainingLogger")
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
