from typing import TextIO
import subprocess
import gzip


def get_lines_count(filepath: str) -> int:
    if filepath.endswith(".gz"):
        command = "cat {0} | gunzip | wc -l".format(filepath)
    else:
        command = "wc -l {0}".format(filepath)
    stdout, _stderr = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       shell=True).communicate()
    return int(stdout.decode("utf-8").split(" ")[0])


def open_file(filepath: str) -> TextIO:
    if filepath.endswith(".gz"):
        return gzip.open(filepath)
    else:
        return open(filepath)
