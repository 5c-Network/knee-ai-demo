import logging

def custom_logger(name=None):
    logger = logging.getLogger(name)

    format = '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s'
    formatter = logging.Formatter(format)

    file_handler = logging.FileHandler('animal_detection.log')
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
