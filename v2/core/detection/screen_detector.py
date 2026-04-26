import torch
import numpy as np
import cv2

class ScreenDetector:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.history = []
        self.sync_config()
        
    def detect(self, frame, stop_ctrl=None):
        self.sync_config()
        
        if hasattr(self.engine, "log"):
            self.engine.log(f"[cfg] side={self.side}, steps={self.steps}, thr={self.threshold}")
            
        h, w, _ = frame.shape
        
        max_conf = 0.0
        max_xy = (0,0)
        
        batch = []
        coords = []
        
        def stopped():
            return stop_ctrl and stop_ctrl.is_paused() 
        
        with torch.no_grad():
            for y in range(0, h - self.side, self.steps):
                if stopped():
                    return frame, {"stopped": True}
                
                for x in range(0, w - self.side, self.steps):
                    if stopped():
                        return frame, {"stopped": True}
                    
                    patch = frame[y:y+self.side, x:x+self.side]
                    
                    patch = cv2.resize(patch, (self.side, self.side))
                    patch = patch.astype("float32")/255.0
                    patch = np.transpose(patch, (2,0,1))
                    
                    tensor = torch.from_numpy(patch)
                    batch.append(tensor)
                    coords.append((x,y))
                    
                    if len(batch) == self.batch_size:
                        max_conf, max_xy = self._process_outputs(
                            batch, coords, frame, self.threshold, max_conf, max_xy
                        )
                        batch.clear()
                        coords.clear()
        
            if batch:
                max_conf, max_xy = self._process_outputs(
                    batch, coords, frame, self.threshold, max_conf, max_xy
                )

        self.history.append(max_conf)
        if len(self.history) > 5:
            self.history.pop(0)

        weights = np.linspace(0.1, 1.0, len(self.history))
        avg_conf = float(np.average(self.history, weights=weights))

        cv2.putText(frame, f"conf={avg_conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if avg_conf >= self.threshold:
            cv2.putText(frame, "detected", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        debug_info = {
            "avg_conf": float(avg_conf),
            "max_conf": float(max_conf),
            "threshold": self.threshold,
            "num_scans": len(self.history),
            "position": max_xy
        }
        
        if hasattr(self.engine, "log"):
            self.engine.log(f"[detect] {debug_info}")
        
        return frame, {
            "class": 1 if max_conf >= self.threshold else 0,
            "confidence": float(max_conf),
            "avg_conf": float(avg_conf),
            "max_conf": float(max_conf),
            "position": max_xy
        }
        
    def _process_outputs(self, batch, coords, frame, threshold, max_conf, max_xy):
        batch_tensor = torch.stack(batch).to(self.engine.device)
        out = self.engine.forward(batch_tensor)

        if out.shape[1] == 1:
            probs = torch.sigmoid(out).squeeze(1).cpu().numpy()
        else:
            probs = torch.softmax(out, dim=1).cpu().numpy()

        for i, (x, y) in enumerate(coords):
            conf = float(probs[i]) if probs.ndim == 1 else float(np.max(probs[i]))

            if conf > max_conf:
                max_conf = conf
                max_xy = (x, y)

            if conf >= threshold:
                cv2.rectangle(frame, (x, y), (x+self.side, y+self.side), (0, 255, 0), 2)

        return max_conf, max_xy
    
    def sync_config(self):
        self.side = self.cfg.get_value("inference", "side_len")
        self.steps = self.cfg.get_value("inference", "steps")
        self.threshold = self.cfg.get_value("inference", "threshold")
        self.batch_size = self.cfg.get_value("inference", "batch_size")
        if hasattr(self, "_last_threshold"):
            if self.threshold != self._last_threshold:
                self.history.clear()
        self._last_threshold = self.threshold