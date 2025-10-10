# susie cnn v1
# plan is to fix cnn quirks(not detecting susie well)

# modifiable variables:
# ln 187 if threshold
# ln 60, 122 if generations
# ln 62, 122 if learn rate
# ln 110 if augment count

import numpy as np
import torch, os, time, cv2, mss
import shared
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# custom imports(shared do window input, preproc is preproc)
from shared import outputImgP
from preproc import image_proc, new_augment

# STRICTLY 128 for now(inputs are 200 max)
class sussyCNN(nn.Module):
    def __init__(self, input_size:int=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # hotfix
        with torch.no_grad():
            dummy = torch.zeros(1,3,input_size, input_size)
            flat_size = self.features(dummy).view(1,-1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for gen in range(generations):
        for batchX, batchY in loader:
            optimizer.zero_grad()
            predictions = model(batchX)
            loss = criterion(predictions, batchY)
            loss.backward()
            optimizer.step()
            
        print(f'Gen: {gen+1}/{generations} // loss: {loss.item():.5f}')
    print('train done!')
    return model

# le save n load
def save(model, path='C:/Users/Dave/susieML/models/susieCNN.pth'):
    torch.save(model.state_dict(), path)
    print(f"model saved ({path})")
def load(model, path='C:/Users/Dave/susieML/models/susieCNN.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print("model loaded")
    return model

def predict(model, img):
    x = torch.tensor(img/ 255, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    with torch.inference_mode():
        out = model(x)
        conf = out.item()
    print(f"confidence level: {conf:.3f}")
    return conf

def main(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model = sussyCNN().to(device)
    if mode == '3':
        train_path = outputImgP()
        if isinstance(train_path, tuple):
            train_path = train_path[0]
        if len(train_path) == 0:
            print("no files :(")
            exit() 
        train_labels = [[1] if 'pos' in p.lower() else [0] for p in train_path]
        print("Training paths: ", train_path)
        print("Labels: ", train_labels)

        train_input, checker_train = new_augment(train_path, train_labels, augment_count=10, mode='cnn')
        epsilon = 0.05
        checker_train = checker_train * (1-epsilon)+(epsilon/2)
        train_input = np.array(train_input).reshape(-1,3,128,128)
        checker_train = np.array(checker_train).reshape(-1,1)
        
        X_train = torch.tensor(train_input, dtype=torch.float32).to(device)
        y_train = torch.tensor(checker_train, dtype=torch.float32).to(device)
        
        model = sussyCNN().to(device)
        lossFN = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
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
        model = sussyCNN().to(device)
        torch.save(model.state_dict(),'C:/Users/Dave/susieML/models/susieCNN.pth')
        print("model saved to models folder.")
    
    # cnn detection part    
    elif mode == '4':
        modelPath = 'C:/Users/Dave/susieML/models/susieCNN.pth'
        cooldown = 1
        sideLen = 128 # input from image
        steps = 96 # scanner-type steps
        windowHistory = 5
        detectionsHistory = []
        
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"model FNF: {modelPath}")
        print("model/s loaded with input size well")
        
        sct = mss.mss()
        monitor = sct.monitors[1]
        lastDetectionTime = 0
        while True:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            # reset per frame vars
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            countmap = np.zeros_like(heatmap)
            max_conf = 0.0
            max_xy = (0,0)
            detectedThisFrame = False
            
            # window scanning
            for y in range(0, frame.shape[0] - sideLen, steps):
                for x in range(0, frame.shape[1]-sideLen, steps):
                    patch = frame[y:y+sideLen, x:x+sideLen]
                    tensor = (torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float() / 255).to(device)
                    tensor = tensor.to(device)
                    # predict but cnn
                    with torch.no_grad():
                        pred = torch.sigmoid(model(tensor)).item()
                    if pred > max_conf:
                        max_conf = pred
                        max_xy = (x,y)
                    heatmap[y:y+sideLen, x:x+sideLen] += pred
                    heatmap = np.clip(heatmap,0,1)
                    countmap[y:y+sideLen, x:x+sideLen] += 1
            
            heatmap /= np.maximum(countmap, 1)
            threshold = 0.67 # 0.67 default(cuz why not)
            
            mask = heatmap >= threshold
            y, x = np.where(mask)
            # shows average heatmap box
            if len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0),2)
            
            # detection history... duh(prints only in consistent frames)
            detectionsHistory.append(max_conf)
            if len(detectionsHistory) > windowHistory:
                detectionsHistory.pop(0)
            avg_conf = np.mean(detectionsHistory)

            # confidence decay floor fix
            current_time = time.time()
            if avg_conf >= threshold and current_time - lastDetectionTime > cooldown:
                lastDetectionTime = current_time
                detectedThisFrame = True
                x,y = max_xy
                cv2.rectangle(frame, (x,y), (x+sideLen, y+sideLen), (0,255,0),2)
                print(f"sussy maybe found..? average conf lvl: {max_conf:.3f}, max conf lvl:{max_conf:.3f}")
            else: print(f"no sussy :( avg=({avg_conf:.3f}), max=({max_conf:.3f}) ")  
            
            cv2.imshow("bleh twan detection(CNN)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        cv2.destroyAllWindows()
    else: print('Invalid mode..')