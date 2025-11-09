# susie cnn v1.01
# fixed confidence output issues
# plan is to add app-like support

# modifiable variables:
# ln 153 if threshold
# ln 67, 135 if generations
# ln 67, 133 if learn rate
# ln 122 if augment count
# ln 87, 90, 145, 150 if save path

import numpy as np
import torch, os, cv2, mss
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# custom imports(shared do window input, preproc is preproc)
from shared import outputImgP, savePath, loadPath
from preproc import new_augment

# the brain; input = 128x128x3
class sussyCNN(nn.Module):
    def __init__(self, input_size:int=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        # hotfix
        with torch.no_grad():
            dummy = torch.zeros(1,3,input_size, input_size)
            flat_size = self.features(dummy).view(1,-1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        x = self.features(x)
        flat = x.view(x.size(0),-1)
        x = self.classifier(flat)
        return x

# ts combine pos and neg images(BCE for now)
def prepareData(pos, neg):
    X = np.concatenate([pos, neg], axis=0).astype(np.float32) / 255
    y = np.concatenate([
        np.ones((len(pos), 1), dtype=np.float32),
        np.zeros((len(neg), 1), dtype=np.float32)
    ])
    X = torch.tensor(X).permute(0,3,1,2)
    y = torch.tensor(y)
    dataset = TensorDataset(X, y)
    # batch size modifier
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    return loader

def train(model, loader, generations=50, lr=5e-4):
    # bce equation got 1-lined... which is good
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for gen in range(generations):
        epoch_loss = 0.0
        for batchX, batchY in loader:
            optimizer.zero_grad()
            predictions = model(batchX)
            loss = criterion(predictions, batchY)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/len(loader)    
        print(f'Gen: {gen+1}/{generations} // loss: {avg_loss:.5f}')
    print('train done!')
    return model

# le save n load
# gonna add the dynamic pathing for convenience
def save(model, path=None):
    torch.save(model.state_dict(), path)
    print(f"model saved ({path})")
def load(model, path=None):
    model.load_state_dict(torch.load(path))
    model.eval()
    print("model loaded")
    return model

def predict(model, img):
    x = torch.tensor(img/ 255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x)
        conf = torch.sigmoid(logits).item()
    print(f"confidence level: {conf:.3f}")
    return conf

def main(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model = sussyCNN().to(device)
    if mode == '3':
        modelSave = savePath()
        # file picker window
        train_path = outputImgP() # window picker from 
        if isinstance(train_path, tuple):
            train_path = train_path[0]
        if len(train_path) == 0:
            print("no files :(")
            exit() 
        train_labels = [[1] if 'pos' in p.lower() else [0] for p in train_path]
        print("Training paths: ", train_path)
        print("Labels: ", train_labels)
        
        train_input, checker_train = new_augment(train_path, train_labels, augment_count=5, mode='cnn')
        epsilon = 0.05
        checker_train = checker_train * (1-epsilon)+(epsilon/2)
        
        train_input = np.array(train_input).reshape(-1,3,128,128)
        checker_train = np.array(checker_train).reshape(-1,1)
        
        X_train = torch.tensor(train_input, dtype=torch.float32).to(device)
        y_train = torch.tensor(checker_train, dtype=torch.float32).to(device)
        
        lossFN = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        # training loop..!
        generations=50
        for gen in range(generations):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = lossFN(outputs, y_train)
            loss.backward()
            optimizer.step()
            if gen % 1 == 0:
                print(f"Gen: {gen+1}/{generations} ; loss: {loss.item():.5f}")
        
        # save model n input
        torch.save(model.state_dict(), modelSave) # input full save path
        print("model saved to models folder.")
    
    # cnn detection part    
    elif mode == '4':
        torch.set_num_threads(os.cpu_count() or 4)
        load_model = loadPath() # input save path
        threshold = 0.67 # 0.67 default(cuz why not)
        sideLen = 128 # scan window size
        steps = 96 # scanner steps(64 fast, 32 depth)
        window_history = 5
        detections_history = []
        
        # checks if path exists..
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"model FNF: {load_model}")
        print("model/s loaded with input size well")
        
        model = sussyCNN().to(device)
        model.load_state_dict(torch.load(load_model, map_location=device))
        model.eval() # no no train
        
        # monitor setup
        sct = mss.mss()
        monitor = sct.monitors[1]
        while True:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            # reset per frame vars
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            countmap = np.zeros_like(heatmap)
            patches = []
            coords = []
            max_conf = 0.0
            max_xy = (0,0)
            
            # window scanning
            for y in range(0, frame.shape[0] - sideLen, steps):
                for x in range(0, frame.shape[1]-sideLen, steps):
                    patch = frame[y:y+sideLen, x:x+sideLen]
                    color_patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2BGR)
                    permuted = torch.tensor(color_patch/255, dtype=torch.float32).permute(2,0,1)
                    patches.append(permuted)
                    coords.append((x,y))
                    
            # prediction part maybe
            batch = torch.stack(patches).to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(batch)).cpu().numpy().flatten()
            
            # building heatmap
            for (x,y), pred in zip(coords, preds):
                conf = float(np.squeeze(pred))
                heatmap[y:y+sideLen, x:x+sideLen] += conf
                countmap[y:y+sideLen, x:x+sideLen] += 1
                if conf > max_conf:
                    max_conf = conf
                    max_xy = (x,y)
             
            # shows average confidence points
            heatmap /= np.maximum(countmap, 1)
            mask = heatmap >= threshold
            y, x = np.where(mask)
            if len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0),2)
            
            # detection history 
            max_conf = np.max(preds)
            detections_history.append(max_conf)
            if len(detections_history) > window_history:
                detections_history.pop(0)
            avg_conf = np.mean(detections_history)

            # confidence decay floor fix
            # detects the most certain object and boxes it out
            if avg_conf >= threshold:
                x,y = max_xy
                cv2.rectangle(frame, (x,y), (x+sideLen, y+sideLen), (0,255,0),2)
                print(f"sussy maybe found..? average conf lvl: {max_conf:.3f}, max conf lvl:{max_conf:.3f}")
            else:
                print(f"no sussy :( avg=({avg_conf:.3f}), max=({max_conf:.3f}) ")  
            preview = cv2.resize(frame, (480,270))
            cv2.imshow("bleh twan detection(CNN)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        cv2.destroyAllWindows()

    else: print('Invalid mode..')

