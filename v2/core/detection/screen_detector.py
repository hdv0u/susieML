import torch
import numpy as np
import cv2
from data.preproc import preprocess

class ScreenDetector:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.history = []
        self.side = self.cfg["inference"]["side_len"]
        
    def detect(self, frame):
        steps = self.cfg["inference"]["steps"]
        threshold = self.cfg["inference"]["threshold"]
        batch_size = self.cfg["inference"].get("infer_batch", 4)
        
        h, w, _ = frame.shape
        
        max_conf = 0.0
        max_xy = (0,0)
        
        batch = []
        coords = []
        
        logits_collected = []
        
        with torch.no_grad():
            for y in range(0, h - self.side, steps):
                for x in range(0, w - self.side, steps):
                    patch = frame[y:y+self.side, x:x+self.side]
                    
                    patch = cv2.resize(patch, (self.side, self.side))
                    patch = patch.astype("float32")/255.0
                    patch = np.transpose(patch, (2,0,1))
                    
                    tensor = torch.from_numpy(patch)
                    batch.append(tensor)
                    coords.append((x,y))
                    
                    if len(batch) == batch_size:
                        max_conf, max_xy = self._process_outputs(
                            batch, coords, frame, threshold, max_conf, max_xy, logits_collected
                        )
                        batch.clear()
                        coords.clear()
        
            if batch:
                max_conf, max_xy = self._process_outputs(
                    batch, coords, frame, threshold, max_conf, max_xy
                )

        self.history.append(max_conf)
        if len(self.history) > 5:
            self.history.pop(0)

        weights = np.linspace(0.1, 1.0, len(self.history))
        avg_conf = float(np.average(self.history, weights=weights))

        cv2.putText(frame, f"conf={avg_conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        debug_info = {
            "avg_conf": float(avg_conf),
            "max_conf": float(max_conf),
            "threshold": threshold,
            "num_scans": len(self.history),
            "position": max_xy
        }
        
        if hasattr(self.engine, "log"):
            self.engine.log(f"[detect] {debug_info}")
        
        return frame, {
            "class": 1 if max_conf >= threshold else 0,
            "confidence": float(avg_conf),
            "max_conf": float(max_conf),
            "position": max_xy
        }
        
    def _process_outputs(self, batch, coords, frame, threshold, max_conf, max_xy, logits_collected):
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