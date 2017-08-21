# -*- coding: utf-8 -*-

def write_lines(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write("%s\n" % line.encode("utf-8"))
