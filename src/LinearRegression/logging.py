import logging


def get_logger():
    
    logging.captureWarnings(True)
    logger = logging.getLogger()
    
    logger.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s] [%(levelname)s]: %(message)s"
    formatter = logging.Formatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    return logger