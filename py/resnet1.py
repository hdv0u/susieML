import torch, cv2, mss, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
from shared import outputImgP, savePath, loadPath
from preproc import new_augment

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 
               
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class sussyResNet(nn.Module):
    def __init__(self, layers=[3,4,6,3], num_class=1) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1],stride=2)
        self.layer3 = self._make_layer(256, layers[2],stride=2)
        self.layer4 = self._make_layer(512, layers[3],stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Bottleneck.expansion, num_class)
        
    def _make_layer(self, mid_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels*Bottleneck.expansion,1,stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels*Bottleneck.expansion)
            )
        
        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = mid_channels * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        

def main(mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None):
    multi_class = False
    num_class = 1
    
    if multi_class:
        num_class = int(input("class count: "))
    device = "cuda" if torch.cuda.is_available() else "cpu"; log_fn(f"using {device}")
    model = sussyResNet(layers=[3,4,6,3], num_class=num_class).to(device)
    # train
    if mode == "5":
        modelSave = savePath()
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
        y_train = torch.tensor(checker_train.reshape(-1,1), dtype=torch.float32).to(device)
        
        if multi_class:
            lossFN = nn.CrossEntropyLoss()
        else: lossFN = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        
        # training loop..!
        batch_size = 16
        generations = 50
        num_samples = X_train.size(0)
        for gen in range(generations):
            model.train()
            epoch_loss = 0.0
            
            indices = torch.randperm(num_samples)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                batch_y = batch_y.float()
                
                loss = lossFN(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
            avg_loss = epoch_loss / num_samples
            if gen % 1 == 0:
                log_fn(f"Gen: {gen+1}/{generations} ; loss: {avg_loss:.5f}")
        
        # save model n input
        torch.save(model.state_dict(), modelSave) # input full save path
        log_fn("model saved to models folder.")
    
    # detect(load)
    elif mode == "6":
        torch.set_num_threads(os.cpu_count() or 4)
        load_model = loadPath() # input save path
        threshold = 0.67 # 0.67 default(cuz why not)
        sideLen = 128 # scan window size
        steps = 64 # scanner steps(64 fast, 32 depth)
        window_history = 5
        detections_history = []
        
        # checks if path exists..
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"model FNF: {load_model}")
        log_fn("model/s loaded with input size well")
        
        model = sussyResNet().to(device)
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
                log_fn(f"sussy maybe found..? average conf lvl: {max_conf:.3f}, max conf lvl:{max_conf:.3f}")
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
    else: log_fn("Invalid input cro.")