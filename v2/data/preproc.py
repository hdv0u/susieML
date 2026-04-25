import cv2, random, torch
import numpy as np

# screen recorder(for future dsand runs)
# idk where for now

# image processor(for screenshot test runs)
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"'I can't read {path}' - person when reading a non-img file.")
    return img

def preprocess(img, cfg):
    # resize and normalize
    img = cv2.resize(img, tuple(cfg["model"]["input_size"]))
    img = img.astype('float32') / 255.0
    
    img = np.transpose(img, (2,0,1))
    return img

def process_image(path, cfg, augment=False):
    img = load_image(path)
    if augment:
        img = apply_augment(img, cfg["augment"])
    img = preprocess(img, cfg)
    return img

def apply_augment(img, aug_cfg):
    # horizontal flip
    if aug_cfg.get("flip_enabled", True) and random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # brightness augmentation
    if aug_cfg.get("brightness_enabled", True) and random.random() > 0.5:
        b_min = aug_cfg.get("brightness_min", 0.8)
        b_max = aug_cfg.get("brightness_max", 1.2)
        factor = b_min + random.random() * (b_max - b_min)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    
    # shift/translation augmentation
    if aug_cfg.get("shift_enabled", True) and random.random() > 0.75:
        height, width = img.shape[:2]
        maxShift = int(aug_cfg.get("shift_max", 0.1) * min(height, width))
        tx = random.randint(-maxShift, maxShift)
        ty = random.randint(-maxShift, maxShift)
        m = np.float32([[1,0,tx], [0,1,ty]])
        img = cv2.warpAffine(img, m, (width, height))
        
    return img

def build_dataset(paths, labels, cfg):  
    xs, ys = [],[]
    
    augment_count = cfg["augment"]["augment_count"]
    
    for path, label in zip(paths, labels):
        xs.append(process_image(path, cfg, augment=False))  
        ys.append(label)
        
        for _ in range(augment_count):
            xs.append(process_image(path, cfg, augment=True))
            ys.append(label)
    
    # stack inputs
    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    # labels
    y = torch.tensor(np.array(ys), dtype=torch.float32).reshape(-1,1)
    return x, y

# consistency with augment n shi
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)