import time

def get_execution_time(function, *args):
    time_start = time.time()
    returned_value = function(*args)
    time_end = time.time()
    return (time_end - time_start, returned_value)
