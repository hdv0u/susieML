import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import DataLoader
from core.dataset import NumpyDataset

# config.py
from config import DEVICE, BATCH_SIZE, GENERATIONS, LEARNING_RATE, CNN

# main.py
from main import DEBUG
print(f"debug: widget {__name__}")

class CNNTrainer:
    def __init__(self, model_fn, device, cfg=None, log_fn=print, progress_fn=None, stop_fn=None, multi_class=False):
        self.model_fn = model_fn
        self.device = device or DEVICE
        self.cfg = cfg or {}
        self.log = log_fn
        self.progress = progress_fn
        self.stop_fn = stop_fn
        self.multi_class = multi_class

    def train_from_paths(self, train_paths, save_path, augment_fn, label_from_path=None, labels=None):
        # prepare model
        model = self.model_fn().to(self.device)
        # derive labels
        if labels is None:
            if self.multi_class:
                raise ValueError("multi-class requires labels from gui")
            label_from_path = label_from_path or (lambda p: 1 if 'pos' in p.lower() else 0)
            labels = [label_from_path(p) for p in train_paths]
        self.log("Training paths:", train_paths)
        self.log("Labels:", labels)

        label_mode = 'cce' if self.multi_class else 'bce'
        
        # augmentation + prepare arrays
        augment_count = self.cfg.get("augment_count", 5)
        train_input, checker_train = augment_fn(train_paths, labels, augment_count=augment_count,
                                                mode='cnn', label_mode=label_mode)
        
        if self.multi_class:
            y = np.array(checker_train).astype(np.int64)
            label_smoothing = self.cfg.get("label_smoothing", 0.0)
            label_mode = 'cce'
            loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            epsilon = self.cfg.get("epsilon", 0.05)
            checker_train = checker_train * (1 - epsilon) + (epsilon / 2)
            y = np.array(checker_train).reshape(-1, 1)
            label_mode = 'bce'
            loss_fn = nn.BCEWithLogitsLoss()

        X = np.array(train_input).reshape(-1, 3, self.cfg.get("side_len", 128), self.cfg.get("side_len", 128))
        
        batch_size = self.cfg.get("batch_size", 16)
        dataset = NumpyDataset(X, y, multi_class=self.multi_class)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.get("lr", 5e-4))
        generations = self.cfg.get("generations", 50)

        for gen in range(generations):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                if self.stop_fn and self.stop_fn():
                    self.log("Training stopped by user")
                    return

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                
                if self.multi_class:
                    loss = loss_fn(outputs, y_batch)
                else:
                    loss = loss_fn(outputs, y_batch.float())
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(loader))
            if self.progress:
                percent = int((gen + 1) / generations * 100)
                self.progress(percent)
            self.log(f"Gen: {gen+1}/{generations} ; loss: {avg_loss:.5f}")

        torch.save(model.state_dict(), save_path)
        self.log(f"Model saved to {save_path}")

class CNNInference:
    def __init__(self, model, device, cfg, log_fn=print, multi_class=False, num_classes=None) -> None:
        self.model = model
        self.device = device or DEVICE
        self.cfg = cfg or CNN
        self.log = log_fn
        self.multi_class = multi_class
        self.num_classes = num_classes if multi_class else 1
        self.model.eval()
        self.detections_history = []
        
    def step(self, frame):
        # ensure frame is 3 channels (BGR)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise ValueError(f"Expected 3-channel frame, got {frame.shape[2]} channels")

        if 'side_len' not in self.cfg:
            return self._dense_step(frame)
        
        return self._cnn_step(frame)
    
    def _dense_step(self, frame):
        h, w, _ = frame.shape
        img = cv2.resize(frame, (128,128))
        tensor = torch.tensor(img / 255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            if self.multi_class:
                preds = torch.softmax(out, dim=1).cpu().numpy()[0]
            else:
                preds = torch.sigmoid(out).cpu().numpy()[0]
                
        conf_val = max(preds) if self.multi_class else float(preds[0])
        
        self.detections_history.append(conf_val)
        if len(self.detections_history) > 5:
            self.detections_history.pop(0)
        avg_conf = sum(self.detections_history)/len(self.detections_history)
        
        cv2.putText(frame, f"conf={avg_conf:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if avg_conf >= self.cfg.get('threshold', 0.5):
            cv2.putText(frame, "detected", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        return frame
    
    def _cnn_step(self, frame):
        side = self.cfg["side_len"]
        steps = self.cfg["steps"]
        threshold = self.cfg["threshold"]
        window_history = self.cfg.get("window_history", 5)
        
        h, w, _ = frame.shape
        patches = []
        coords = []
        
        # window scanning
        for y in range(0, h - side, steps):
            for x in range(0, w - side, steps):
                patch = frame[y:y+side, x:x+side]
                if patch.shape[:2] != (side, side):
                    patch = cv2.resize(patch, (side,side))
                tensor = torch.tensor(patch / 255., dtype=torch.float32).permute(2,0,1)
                patches.append(tensor)
                coords.append((x,y))
                
        if not patches: return frame
        
        # prediction part
        batch_size = self.cfg.get("infer_batch", 8)
        preds = []
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(self.device)
                if self.multi_class:
                    out = torch.softmax(self.model(batch), dim=1)
                else:
                    out = torch.sigmoid(self.model(batch))
                preds.extend(out.cpu())
        
        preds = torch.stack(preds, dim=0).numpy()
        
        heatmaps = [np.zeros((h,w), dtype=np.float32) for _ in range(self.num_classes if self.multi_class else 1)]
        countmap = np.zeros_like(heatmaps[0])
        max_conf = 0.0
        max_xy = (0,0)   
        
        for idx, (x,y) in enumerate(coords):
            if self.multi_class:
                for cls in range(self.num_classes):
                    conf_val = preds[idx][cls]
                    heatmaps[cls][y:y+side, x:x+side] += conf_val
                    if conf_val > max_conf:
                        max_conf = conf_val
                        max_xy = (x,y)
            else:
                val = preds[idx]
                if isinstance(val, (np.ndarray, list)):
                    conf_val = float(val[0])
                else:
                    conf_val = float(val)
                heatmaps[0][y:y+side, x:x+side] += conf_val
                if conf_val > max_conf:
                    max_conf = conf_val
                    max_xy = (x,y)    
            countmap[y:y+side, x:x+side] += 1
        
        for i in range(len(heatmaps)):
            heatmaps[i] /= np.maximum(countmap,1)
            
        self.detections_history.append(max_conf)
        if len(self.detections_history) > window_history:
            self.detections_history.pop(0)    
        avg_conf = sum(self.detections_history)/len(self.detections_history)
            
        x,y = max_xy
        cv2.rectangle(frame, (x,y), (x+side, y+side), (0,255,0), 2)
        cv2.putText(frame, f"conf={avg_conf:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        if avg_conf >= threshold:
            cv2.putText(frame, "detected", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        self.log(f"detection avg/max={max_conf:.3f}")  
        
        return frame