import logging
import time

logging.basicConfig(level=logging.INFO)


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"Execution time ({func.__name__}): {time.time() - start:.4f} sec")
        return result

    return wrapper
