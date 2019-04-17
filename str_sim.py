from os import listdir
from difflib import SequenceMatcher


def str_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def search_filename(srchstr,dirname):
    ss = [x for x in listdir(dirname)]
    sm = 0
    ms = None
    for s in ss:
        m = str_sim(s,srchstr)
        if sm < m:
            sm,ms = m,s
    return ms
