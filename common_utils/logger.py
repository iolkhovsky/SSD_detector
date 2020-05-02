import time
from common_utils.timestamp import  get_readable_timestamp


class LogDuration:

    def __init__(self, hint=None):
        self.start_timestamp = time.time()
        return

    def get(self):
        current = time.time()
        return current - self.start_timestamp


class Logger:

    def __init__(self, path=None, hint=None):
        self.log_file = "log.txt"
        if path:
            self.log_file=path
        if hint:
            self.log_file = hint+"_"+self.log_file
        return

    def __call__(self, *args, **kwargs):
        buf = get_readable_timestamp() + " ";
        if "caller" in kwargs.keys():
            buf += "<" + kwargs["caller"] + ">"
        buf += ": "
        for arg in args:
            buf += str(arg) + " "
        print(buf)
        with open(self.log_file, "a") as f:
            f.write(buf)
        return

    def log_dict(self, d, **kwargs):
        caller = None
        if "caller" in kwargs.keys():
            caller = kwargs["caller"]
        if type(d) == dict:
            for k in d.keys():
                if caller:
                    self.__call__(k, ": ", d[k], caller=caller)
                else:
                    self.__call__(k, ": ", d[k])

