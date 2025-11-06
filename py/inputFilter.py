import re, os, numpy as np
# add window shi instead of copypasting file(done 10/6/25)
# not fully retired tho
def getImage(raw_input, extensions=None):
    if isinstance(raw_input, bytes):
        raw_input = raw_input.decode('utf-8', errors='ignore')
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    tokens = re.split(r"[,\s;|]+", raw_input.strip())
    seen = set()
    clean_paths = []
    for t in tokens:
        if not t: continue
        t = t.strip().strip("'\"")
        ext = os.path.splitext(t)[1].lower()
        if ext not in extensions:
            continue
        t_clean = t.replace("\\", "/").lower()
        if t_clean in seen:
            continue
        seen.add(t_clean)
        clean_paths.append(t_clean) # 1d list
    return np.array(clean_paths, dtype=str)

def getLabel(paths_array, keyword="pos"):
    keyword = keyword.lower()
    labels = []
    for path in paths_array:
        parts = re.split(r"[\\/]", path)
        label = 1 if keyword in parts else 0
        labels.append(label)
    return np.array(labels, dtype=int)

def ppMatrix(raw_input):
    X = getImage(raw_input)
    y = getLabel(X, keyword="pos")
    return X, y
