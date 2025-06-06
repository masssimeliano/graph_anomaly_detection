import logging
import time

logging.basicConfig(level=logging.INFO)


# measures time for given function and logs timing
def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logging.info(f"Execution time ({func.__name__}): {(time.time() - start_time):.4f} seс")
        return result

    return wrapper
