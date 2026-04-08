import cv2, csv, random, torch
import numpy as np
import config
# screen recorder(for future dsand runs)
# see torchML at elif == 4

# image processor(for screenshot test runs)
def image_proc(input_data, size=(128,128), augment=False, debug=False, 
                mode='dense', normalize=True, mean=None, std=None, augment_cfg=None):
    def single_proc(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"'I can't read {path}' - person when reading a non-img file.")
        # augmentation for image multiplication
        if augment and augment_cfg:
            # horizontal flip
            if augment_cfg.get("flip_enabled", True) and random.random() > 0.5:
                img = cv2.flip(img, 1)
            # brightness augmentation
            if augment_cfg.get("brightness_enabled", True) and random.random() > 0.5:
                b_min = augment_cfg.get("brightness_min", 0.8)
                b_max = augment_cfg.get("brightness_max", 1.2)
                factor = b_min + random.random() * (b_max - b_min)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
            # shift/translation augmentation
            if augment_cfg.get("shift_enabled", True) and random.random() > 0.75:
                height, width = img.shape[:2]
                shift_factor = augment_cfg.get("shift_max", 0.1)
                maxShift = int(shift_factor * min(height, width))
                tx = random.randint(-maxShift, maxShift)
                ty = random.randint(-maxShift, maxShift)
                m = np.float32([[1,0,tx],[0,1,ty]])
                img = cv2.warpAffine(img, m, (width, height))
        # fallback: basic augmentation if config not provided
        elif augment:
            # 50% horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                factor = 0.8 + random.random() * 0.4 # 80 to 120% bright
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
            # 25% translation
            if random.random() > 0.75:
                height, width = img.shape[:2]
                maxShift = int(0.1 * min(height, width))
                tx = random.randint(-maxShift, maxShift)
                ty = random.randint(-maxShift, maxShift)
                m = np.float32([[1,0,tx],[0,1,ty]])
                img = cv2.warpAffine(img, m, (width, height))
        # resize and normalize
        img = cv2.resize(img, size)
        img = img.astype('float32') / 255.0 if normalize else img.astype('float32')
        # output image
        if mean is not None and std is not None:
            meanArr = np.array(mean).reshape(1,1,3)
            stdArr = np.array(std).reshape(1,1,3)
            img = (img - meanArr) / stdArr
            
        if debug:
            cv2.imshow("print success!!!!!", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if mode == 'dense':
            return img.flatten()
        elif mode == 'cnn':
            return np.transpose(img, (2,0,1))
        else: raise ValueError("mode must be dense or cnn...")
    
    # handling paths
    if isinstance(input_data, str):
        return np.expand_dims(single_proc(input_data), axis=0) # single ML food(not recommended for training)
    elif isinstance(input_data, (list, tuple)):
        return np.array([single_proc(p) for p in input_data]) # multiple ML food(good shi)
    else: raise TypeError('must be str or list cro')

def new_augment(paths, labels, augment_count=5, size=(128,128), 
                mode='dense', normalize=True, mean=None, std=None, 
                debug=False, return_torch=False, label_mode='auto', augment_cfg=None):
    # very frustrating but its here for a reason
    if len(paths) == 0 or len(labels) == 0:
        raise ValueError("path/label is none. fix input before augment")
    
    # use provided augment_count from config, fallback to parameter
    if augment_cfg and "augment_count" in augment_cfg:
        actual_augment_count = augment_cfg["augment_count"]
    else:
        actual_augment_count = augment_count
    
    xs, ys = [],[]
    for path, label in zip(paths, labels):
        original = image_proc(path, augment=False, mode=mode, augment_cfg=augment_cfg)
        xs.append(original)
        ys.append(label)
        for _ in range(actual_augment_count):
            xs.append(image_proc(path, augment=True, mode=mode, augment_cfg=augment_cfg))
            ys.append(label)
    
    xArr = np.stack(xs).astype(np.float32)

    if label_mode == 'cce':
        yArr = np.array(ys, dtype=np.int64).reshape(-1)
    elif label_mode == "bce":
        yArr = np.array(ys, dtype=np.float32).reshape(-1,1)
    else: raise ValueError("label must be binary or multi-class")
    
    if return_torch:
        xT = torch.tensor(xArr, dtype=torch.float32)
        if label_mode == 'cce':
            yT = torch.tensor(yArr, dtype=torch.long)
        else: yT = torch.tensor(yArr, dtype=torch.long)
        return xT, yT
    
    return xArr, yArr
    

# yayyyyy csv to numpy(not jpg unfortunately)
# useless block for now
def csv_to_np(csv_path, size=(128,128,3)):
    height, width, color = size
    expected_len = height * width * color
    data = [] # very important
    with open(csv_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            pixels = np.array(row, dtype=np.float32)
            if len(pixels) != expected_len:
                continue
            pixels = pixels / 255.0
            flat = pixels.flatten()
            data.append(flat)
    arr = np.stack(data, axis=0)
    return arr

