import logging
import sys

def get_logger(name):
    """
    Returns a production-ready logger that outputs strictly formatted logs to stdout.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is called multiple times for the same name
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create stdout handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Define a professional, standard format: [TIME] [LEVEL] [LOGGER_NAME] - MESSAGE
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
    return logger
