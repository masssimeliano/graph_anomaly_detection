"""
nx_graph_plotter.py
This file contains timing annotation which measures time of method below annotation.
"""

import logging
import time


def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        logging.info(
            f"Execution time ({func.__name__}): {(time.time() - start_time):.4f} seс"
        )
        return result

    return wrapper
