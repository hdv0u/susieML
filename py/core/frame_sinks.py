from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2

def opencv_sink(window_name='susieML v1.0.0'):
    def sink(frame):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration
    return sink

def pyqt_sink(label_widget):
    def sink(frame: np.ndarray):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('frame must be HxWx3 BGR array')
        
        h,w,ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        
        label_widget.setPixmap(pix)
        
        return sink