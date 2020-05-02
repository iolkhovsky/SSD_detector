from os.path import isfile, isdir, join
from os import listdir


def get_files_list(path, filter_key=".xml"):
    if filter_key is None:
        return [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    else:
        return [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and filter_key in f)]


