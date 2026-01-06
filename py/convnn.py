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
# custom imports(shared do window input, preproc is preproc)
from shared import outputImgP, savePath, loadPath
from preproc import new_augment
import config
# the brain; input = 128x128x3
class sussyCNN(nn.Module):
    def __init__(self, input_size:int=128, out_channels=1):
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

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4)),
        )
        # hotfix
        with torch.no_grad():
            dummy = torch.zeros(1,3,input_size, input_size)
            flat_size = self.features(dummy).view(1,-1).size(1)
            
        self.out_channels = out_channels
        self.classifier = nn.LazyLinear(self.out_channels)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class DataSet(torch.utils.data.Dataset):
    def __init__(self,X,y) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def main(mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log_fn(f"using {device}")
    model = sussyCNN().to(device)
    if mode == '3':
        modelSave = savePath()
        # file picker window
        train_path = outputImgP() # window picker from 
        if isinstance(train_path, tuple):
            train_path = train_path[0]
        if len(train_path) == 0:
            log_fn("no files :(")
            exit() 
        train_labels = [[1] if 'pos' in p.lower() else [0] for p in train_path]
        log_fn("Training paths: ", train_path)
        log_fn("Labels: ", train_labels)
        
        train_input, checker_train = new_augment(train_path, train_labels, augment_count=5, mode='cnn')
        epsilon = 0.05
        checker_train = checker_train * (1-epsilon)+(epsilon/2)
        
        train_input = np.array(train_input).reshape(-1,3,128,128)
        checker_train = np.array(checker_train).reshape(-1,1)
        
        X_train = torch.tensor(train_input, dtype=torch.float32).to(device)
        y_train = torch.tensor(checker_train, dtype=torch.float32).to(device)
        
        batch_size = 16

        train_data = DataSet(train_input, checker_train)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        lossFN = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        # training loop..!
        generations=50
        for gen in range(generations):
            model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                if stop_fn and stop_fn():
                    log_fn("train stopped by user")
                    return
                
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(X_batch)
                loss = lossFN(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            if gen % 1 == 0:
                if progress_fn:
                    percent = int((gen + 1) / generations * 100)
                    progress_fn(percent)
                avg_loss = total_loss/len(train_loader)
                log_fn(f"Gen: {gen+1}/{generations} ; loss: {loss.item():.5f}")
        
        # save model n input
        torch.save(model.state_dict(), modelSave) # input full save path
        log_fn("model saved to models folder.")
    
    # cnn detection part    
    elif mode == '4':
        cfg = config.CNN
        torch.set_num_threads(cfg.get("threads") or os.cpu_count())
        load_model = loadPath() # input save path
        threshold = cfg["threshold"] # 0.67 default(cuz why not)
        sideLen = cfg["side_len"] # scan window size
        steps = cfg["steps"] # scanner steps(64 fast, 32 depth)
        window_history = cfg["window_history"]
        detections_history = []
        
        # checks if path exists..
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"model FNF: {load_model}")
        log_fn("model/s loaded with input size well")
        
        model = sussyCNN().to(device)
        model.load_state_dict(torch.load(load_model, map_location=device))
        model.eval() # no no train
        cv2.namedWindow("bleh twan detection(CNN)", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
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
            
            if stop_fn and stop_fn():
                log_fn("train stopped by user")
                return
            
            # window scanning
            for y in range(0, frame.shape[0] - sideLen, steps):
                for x in range(0, frame.shape[1]-sideLen, steps):
                    patch = frame[y:y+sideLen, x:x+sideLen]
                    permuted = torch.tensor(patch/255, dtype=torch.float32).permute(2,0,1)
                    patches.append(permuted)
                    coords.append((x,y))
                    
            # prediction part maybe
            #batch = torch.stack(patches).to(device)
            #with torch.no_grad():
            #    preds = torch.sigmoid(model(batch)).cpu().numpy().flatten()
            
            infer_batch_size = 8
            preds = []
            
            with torch.no_grad():
                for i in range(0, len(patches), infer_batch_size):
                    batch_patches = patches[i:i+infer_batch_size]
                    batch = torch.stack(batch_patches).to(device)
                    
                    batch_preds = torch.sigmoid(model(batch))
                    preds.extend(batch_preds.cpu().numpy().flatten())
                    
            preds = np.array(preds)
            
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
                log_fn(f"sussy maybe found..? avg=({max_conf:.3f}), max=({max_conf:.3f})")
            else:
                log_fn(f"no sussy :( avg=({avg_conf:.3f}), max=({max_conf:.3f}) ")  
            preview = cv2.resize(frame, (1280,720))
            
            if frame_fn:
                frame_fn(preview)
            else:
                cv2.imshow("bleh twan detection(CNN)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        cv2.destroyAllWindows()

    else: log_fn('Invalid mode..')

