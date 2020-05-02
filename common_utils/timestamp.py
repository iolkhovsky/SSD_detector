import datetime


def get_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return stamp