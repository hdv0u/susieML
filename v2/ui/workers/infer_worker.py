from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
from mss import mss

class InferWorker(QThread):
    frame_ready = pyqtSignal(object)
    log = pyqtSignal(str)
    
    def __init__(self, detector, stop_ctrl):
        super().__init__()
        self.detector = detector
        self.stop_ctrl = stop_ctrl
        
        self._frame_count = 0
        self._last_conf = None
        
    def stop(self):
        self.stop_ctrl.stop()
        
    def run(self):
        self.log.emit(f"[thread start] id={id(self)}")
        
        sct = mss()
        monitor = sct.monitors[1]
        self.log.emit("Screen thingy started")
        
        while self.stop_ctrl.is_running():
            if self.stop_ctrl.is_paused():
                self.msleep(50)
                continue
            
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            result_frame, result = self.detector.detect(frame, stop_ctrl=self.stop_ctrl)
            
            self.frame_ready.emit((result_frame, result))
            
            self._frame_count += 1
            conf = result.get("confidence", None)
            cls = result.get("class", -1)
            
            if self._frame_count % 10 == 0:
                self.log.emit(f"[alive] frame={self._frame_count}")
                
            if conf is not None:
                if (self._last_conf is None or abs(conf - self._last_conf) > 0.1):
                    self.log.emit(f"[infer] class={cls}, conf={conf:.3f}")
                    self._last_conf = conf