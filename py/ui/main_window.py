from PyQt5.QtGui import (
    QImage, QPixmap
)
import cv2
from datetime import datetime

# debug: print(f"debug: widget {__name__}")
class TestUI:
    def __init__(self):
        from PyQt5.QtWidgets import (
            QWidget, QLabel, QRadioButton, QPushButton,
            QVBoxLayout, QButtonGroup, QTextEdit, QProgressBar, QCheckBox,
        )
        from ui.worker import MLWorker
        from core.registry import MODELS
        from ui.file_dialog import save_model_file, select_model_file, labeled_picker, labeled_picker_multi
        class _W(QWidget):
            pass
        self._widget = _W()
        self._widget.setWindowTitle("SusieML")
        
        # the layout
        layout = QVBoxLayout(self._widget)
        
        self._MODELS = MODELS
        self._select_model_file = select_model_file
        self._save_model_file = save_model_file
        self._labeled_picker = labeled_picker
        self._labeled_picker_multi = labeled_picker_multi
        self._MLWorker = MLWorker
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)
        self.video_label.setStyleSheet("background-color: black;")
        
        self.log = QTextEdit(readOnly=True)
        self.progress = QProgressBar()
        self.status = QLabel("idle")
        self.run_btn = QPushButton("Train/Detect")
        self.stop_btn = QPushButton("Stop")
        self.multi_class_checkbox = QCheckBox("Enable Multi-class")
        self.multi_class_checkbox.setChecked(False)
        
        self.mode_group = QButtonGroup(self._widget)
        for mode_id, model_info in MODELS.items():
            rb = QRadioButton(model_info["label"])
            self.mode_group.addButton(rb, int(mode_id))
            layout.addWidget(rb)
        
        layout.addWidget(self.video_label)
        layout.addWidget(self.log)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.multi_class_checkbox)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)
        
        # button callbacks
        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.connect(self.stop)
        
        self.worker = None
    
    @property
    def _multi_class_enabled(self):
        return self.multi_class_checkbox.isChecked()
    
    def show(self):
        self._widget.show()
        
    def append_log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")
    
    def run(self):
        if self.worker and self.worker.isRunning():
            return
        
        selected_id = self.mode_group.checkedId()
        if selected_id == -1:
            self.log.append("No mode selected")
            return
        
        selected_mode = str(selected_id)
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("running")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        model_save = None
        train_paths = None
        train_labels = None
        load_model = None
        multi_class = False
        class_names = []
        # 1,2 for dense
        train_modes = ('1', '3', '5')
        infer_modes = ('2', '4', '6')
        
        if selected_mode in train_modes:
            if self._multi_class_enabled:
                multi_class = True
                class_names = ['Positive', 'Neutral', 'Negative']
            else:
                multi_class = False
                class_names = ['Positive', 'Negative']
            
            train_paths, labels = self._labeled_picker_multi(class_names=class_names)
            if not train_paths:
                self.log.append("no training images selected")
                self._reset_ui()
                return
            
            if model_save is None:
                model_save = self._save_model_file(parent=self._widget)
                if not model_save:
                    self.log.append('train canceled (no save path)')
                    self._reset_ui()
                    return
                
        elif selected_mode in infer_modes:
            load_model = self._select_model_file(parent=self._widget)
            if not load_model:
                self.log.append("no model selected")
                self._reset_ui()
                return
        
        self.worker = self._MLWorker(
            mode=selected_mode,
            model_save=model_save,
            train_paths=train_paths,
            train_labels=labels if selected_mode in ('1', '3', '5') else None,
            load_model=load_model,
            multi_class=self._multi_class_enabled,
            label_widget=self.video_label,
        )
        self.worker.frame.connect(self.update_frame)
        self.worker.log.connect(lambda s: self.log.append(str(s)))
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_done)
        self.worker.start()
    
    def update_frame(self, frame):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            self.log.append(f"Unexpected channels: {frame.shape[2]}")
            return
            
        h,w,ch = frame.shape
        qimg = QImage(
            frame.data,
            w,h,
            ch * w,
            QImage.Format_BGR888
        )
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.width(), self.video_label.height(), aspectRatioMode=1)
        self.video_label.setPixmap(pix)
        
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("stopping..")
              
    def on_done(self):
        self.status.setText("done")
        self._reset_ui()        
        self._cleanup_worker()
        
    def _reset_ui(self):
        self.status.setText("idle")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def _cleanup_worker(self):
        if self.worker:
            self.worker.wait()
            self.worker = None
            