from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QTextEdit,
    QStackedWidget, QRadioButton, QSpinBox, QCheckBox, QProgressBar, QButtonGroup
)
from PyQt5.QtGui import (
    QImage, QPixmap
)
from PyQt5.QtCore import Qt
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.registry import MODELS
from ui.worker import MLWorker
from ui.file_dialog import save_model_file, select_model_file, labeled_picker, labeled_picker_multi
class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        self.setWindowTitle("SusieML")
        self.setMinimumSize(854, 480)
        self.worker = None
        self.settings = {}
        
        self.save_model_file = save_model_file
        self.select_model_file = select_model_file
        self.labeled_picker = labeled_picker
        self.labeled_picker_multi = labeled_picker_multi
        
        self.menu_window() # Menu
        self.train_window() # Train
        self.run_window() # Run
        self.settings_window() # Settings
        self.settings_logic()
        
    def menu_window(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("index 0")
        
        train_btn = QPushButton("Train")
        run_btn = QPushButton("Run")
        settings_btn = QPushButton("Settings")
        layout.addWidget(label)
        layout.addWidget(train_btn)
        layout.addWidget(run_btn)
        layout.addWidget(settings_btn)
        
        train_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        run_btn.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        settings_btn.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        
        self.stack.addWidget(page)
        
    def train_window(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.m_class_cbox1 = QCheckBox("Enable Multi-class")
        self.log1 = QTextEdit(readOnly=True)
        self.progress1 = QProgressBar()
        self.radio_btns1 = [] # store as list for future use
        self.train_btn = QPushButton("Train")
        self.stop_btn1 = QPushButton("Stop")
        self.back_btn1 = QPushButton("Back")
        self.train_mode_group = QButtonGroup(self)
        layout.addWidget(QLabel("Train"))
        for module, mode_label in MODELS.items():
            rb = QRadioButton(mode_label["label"])
            if int(module) % 2 == 1: # train
                layout.addWidget(rb)
                self.train_mode_group.addButton(rb, int(module))
                self.radio_btns1.append(rb)
        
        layout.addWidget(self.m_class_cbox1)
        layout.addWidget(self.log1)
        layout.addWidget(self.progress1)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.stop_btn1)
        layout.addWidget(self.back_btn1)
        
        self.stack.addWidget(page)    
    
    def run_window(self):
        page = QWidget() 
        main_layout = QVBoxLayout(page)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: black;")
        
        self.m_class_cbox2 = QCheckBox("Enable Multi-class")
        self.run_btn = QPushButton("Run")
        self.radio_btns2 = []
        self.stop_btn2 = QPushButton("Stop")
        self.log2 = QTextEdit(readOnly=True)
        self.log2.setMaximumHeight(100)
        self.back_btn2 = QPushButton("Back")
        self.progress2 = QProgressBar()
        
        controls = QVBoxLayout()
        
        self.mode_group = QButtonGroup(self)
        for module, mode_label in MODELS.items():
            rb = QRadioButton(mode_label["label"])
            if int(module) % 2 == 0: # detect/run
                self.mode_group.addButton(rb, int(module))
                controls.addWidget(rb)
                self.radio_btns2.append(rb)
        controls.addWidget(self.m_class_cbox2)
        controls.addWidget(self.log2)
        controls.addWidget(self.progress2)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.stop_btn2)
        controls.addWidget(self.back_btn2)
        
        main_layout.addWidget(QLabel("Run Window"))
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(controls)
        
        self.stack.addWidget(page)    
    
    def settings_window(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        # threshold
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.00)
        self.threshold.setValue(0.67)
        self.threshold.setSingleStep(0.01)
        # side len
        self.side_len = QSpinBox()
        self.side_len.setRange(16, 1024)
        self.side_len.setValue(128)
        self.side_len.setSingleStep(2)
        # steps
        self.steps = QSpinBox()
        self.steps.setRange(1, 1024)
        self.steps.setValue(96)
        self.steps.setSingleStep(2)
        # model depth(implementing for each architecture soon)
        self.arch_depth = QSpinBox()
        
        self.back_btn3 = QPushButton("Back")
        layout.addWidget(QLabel("Settings"))
        
        layout.addWidget(QLabel("Threshold")) 
        layout.addWidget(self.threshold)
        layout.addWidget(QLabel("Side Len"))
        layout.addWidget(self.side_len)
        layout.addWidget(QLabel("Steps")) 
        layout.addWidget(self.steps)
        layout.addWidget(QLabel("Architecture Depth"))
        layout.addWidget(self.arch_depth)
        layout.addWidget(self.back_btn3) 
        
        self.stack.addWidget(page)    
    
    def settings_logic(self):
        # back to menu buttons
        self.back_btn1.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.back_btn2.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.back_btn3.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        
        # run/stop buttons
        self.train_btn.clicked.connect(self.run)
        self.run_btn.clicked.connect(self.run)
        self.stop_btn1.clicked.connect(self.stop)
        self.stop_btn2.clicked.connect(self.stop)
        
        # setting buttons
        self.threshold.valueChanged.connect(lambda val: self.on_spin_changed("threshold", val)) # name, value
        self.side_len.valueChanged.connect(lambda val: self.on_spin_changed("side len", val))
        self.steps.valueChanged.connect(lambda val: self.on_spin_changed("steps", val))
        self.arch_depth.valueChanged.connect(lambda val: self.on_spin_changed("arch depth", val))
    
    def update_frame(self, frame):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            self.log2.append(f"Unexpected channels: {frame.shape[2]}")
            return
            
        h,w,ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video_label.setPixmap(pix)
    
    @property
    def multi_class_enabled(self):
        return (self.m_class_cbox1.isChecked() or self.m_class_cbox2.isChecked())
    
    def run(self):
        if self.worker and self.worker.isRunning():
            return
        if self.stack.currentIndex() == 1:
            selected_id = self.train_mode_group.checkedId()
        else:
            selected_id = self.mode_group.checkedId()
            
        if selected_id == -1:
            self.log2.append("No mode selected")
            return
        
        selected_mode = str(selected_id)
        self.log2.clear()
        self.progress1.setValue(0)
        self.progress2.setValue(0)
        self.log2.setText("running")
        self.run_btn.setEnabled(False)
        self.stop_btn2.setEnabled(True)
        
        model_save = None
        train_paths = None
        load_model = None
        class_names = []
        # 1,2 for dense
        train_modes = ('1', '3', '5')
        infer_modes = ('2', '4', '6')
        labels = None
        if selected_mode in train_modes:
            class_names = (
                ['Positive', 'Neutral', 'Negative']
                if self.multi_class_enabled
                else ['Positive', 'Negative']
            )
            
            train_paths, labels = self.labeled_picker_multi(class_names=class_names)
            if not train_paths:
                self.log1.append("no training images selected")
                self._reset_ui()
                return
            
            model_save = self.save_model_file(parent=self)
            if not model_save:
                self.log1.append('train canceled (no save path)')
                self._reset_ui()
                return
                
        elif selected_mode in infer_modes:
            load_model = self.select_model_file(parent=self)
            if not load_model:
                self.log2.append("no model selected")
                self._reset_ui()
                return
        
        self.worker = MLWorker(
            mode=selected_mode,
            model_save=model_save,
            train_paths=train_paths,
            train_labels=labels,
            load_model=load_model,
            multi_class=self.multi_class_enabled,
            label_widget=self.video_label,
        )
        target_log = self.log1 if selected_mode in train_modes else self.log2
        target_progress = self.progress1 if selected_mode in train_modes else self.progress2
        
        self.worker.frame.connect(self.update_frame)
        self.worker.log.connect(lambda s: target_log.append(str(s)))
        self.worker.progress.connect(target_progress.setValue)
        self.worker.finished.connect(self.on_done)
        self.worker.start()
    
    # train window part
    def on_radio_toggled(self, index, checked):
        if checked:
            print(f"Radio button {index+1} picked")
    
    def on_checkbox_changed(self, state):
        if state == 2: # 2 for check, 0 is uncheck
            print(True)
        else:
            print(False)
              
    # settings logic part
    def on_spin_changed(self, name, value):
        self.settings[name] = value
        print(self.settings)

    def get_settings(self):
        return {
            "threshold": self.threshold.value(),
            "side_len": self.side_len.value(),
            "steps": self.steps.value(),
            "arch_depth": self.arch_depth.value(),
        }
    
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.log1.setText("stopping..")
            self.log2.setText("stopping..")
              
    def on_done(self):
        self.log1.setText("done")
        self._reset_ui()        
        self._cleanup_worker()
        
    def _reset_ui(self):
        self.log1.setText("idle")
        self.train_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.stop_btn1.setEnabled(False)
        self.stop_btn2.setEnabled(False)
        
    def _cleanup_worker(self):
        if self.worker:
            self.worker.wait()
            self.worker = None