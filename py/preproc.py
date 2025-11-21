import cv2, csv, random, torch
import numpy as np
import shared
# screen recorder(for future dsand runs)
# see torchML at elif == 4

# image processor(for screenshot test runs)
def image_proc(input_data, size=(128,128), augment=False, debug=False, 
                mode='dense', normalize=True, mean=None, std=None):
    def single_proc(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"'I can't read {path}' - person when reading a non-img file.")
        # augmentation for image multiplication
        if augment:
            # 50% horizontal n brightness
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                factor = 0.8 + random.random() * 0.4 # 80 to 120% bright
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
            # 25% rotation n gaussian noise (+- 15 degrees)
            if random.random() > 0.75:
                height, width = img.shape[:2]
                maxShift = int(0.1 * min(height, width))
                tx = random.randint(-maxShift, maxShift)
                ty = random.randint(-maxShift, maxShift)
                m = np.float32([[1,0,tx],[0,1,ty]])
                img = cv2.warpAffine(img, m, (width, height))
            if random.random() > 0.75:
                noise = np.random.normal(0, 8, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
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
                debug=False, return_torch=False, label_mode='auto'):
    # very frustrating but its here for a reason
    if len(paths) == 0 or len(labels) == 0:
        raise ValueError("path/label is none. fix input before augment")
    xs, ys = [],[]
    for path, label in zip(paths, labels):
        original = image_proc(path, augment=False, mode=mode)
        xs.append(original)
        ys.append(label)
        for _ in range(augment_count):
            xs.append(image_proc(path, augment=True, mode=mode))
            ys.append(label)
    
    xArr = np.stack(xs).astype(np.float32)
    
    labelsample = ys[0]
    if label_mode == 'auto':
        if isinstance(labelsample, (list, tuple, np.ndarray)) and np.array(labelsample).ndim >= 1 and np.array(labelsample).size > 1:
            yArr = np.array(ys, dtype=np.float32)
        else: yArr = np.array(ys, dtype=np.float32).reshape(-1,1)
    elif label_mode == "bce":
        yArr = np.array(ys, dtype=np.float32).reshape(-1,1)
    elif label_mode == 'cce':
        yArr = np.array(ys, dtype=np.int64).reshape(-1)
    else: raise ValueError("label must be auto/bce/cce")
    
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

