import json

def load_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    paths = []
    labels = []
    
    for p in data["pos"]:
        paths.append(p)
        labels.append(1)
        
    for n in data["neg"]:
        paths.append(n)
        labels.append(0)
        
    return paths, labels