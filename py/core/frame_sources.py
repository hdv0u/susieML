import mss, numpy as np

def screen_source():
    sct = mss.mss()
    monitor = sct.monitors[1]
    while True:
        img = np.array(sct.grab(monitor))
        yield img