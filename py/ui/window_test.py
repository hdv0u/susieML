from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QTextEdit,
    QStackedWidget, QRadioButton, QSpinBox, QCheckBox, QProgressBar, QButtonGroup
)
from PyQt5.QtGui import (
    QImage, QPixmap
)
from PyQt5.QtCore import Qt
from core.registry import MODELS
from config import update_settings
class TestWindow(QWidget):
    def __init__(self) -> None:
        from ui.file_dialog import save_model_file, select_model_file, labeled_picker, labeled_picker_multi
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
        self.arch_depth = QSpinBox()
        self.arch_depth.setRange(1, 8)
        self.arch_depth.setValue(1)
        self.arch_depth.setSingleStep(1)
        # below is train settings
        # epoch/generations
        self.gen = QSpinBox()
        self.gen.setRange(1, 10000)
        self.gen.setValue(50)
        self.gen.setSingleStep(10)
        # learning rate
        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-6, 1e-1)
        self.lr.setDecimals(6)
        self.lr.setValue(5e-4)
        self.lr.setSingleStep(1e-5)
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
        layout.addWidget(QLabel("Architecture Depth"))
        layout.addWidget(self.arch_depth)
        layout.addWidget(QLabel("Epochs/Generations"))
        layout.addWidget(self.gen)
        layout.addWidget(QLabel("Learning Rate"))
        layout.addWidget(self.lr)
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
        
        self.m_class_cbox2 = QCheckBox("Enable Multi-class(Experimental)")
        self.run_btn = QPushButton("Run")
        self.radio_btns2 = []
        self.stop_btn2 = QPushButton("Stop")
        self.log2 = QTextEdit(readOnly=True)
        self.log2.setMaximumHeight(100)
        self.back_btn2 = QPushButton("Back")
        # threshold for inference
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.00)
        self.threshold.setValue(0.67)
        self.threshold.setSingleStep(0.01)
        
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
        controls.addWidget(QLabel("Detection Threshold"))
        controls.addWidget(self.threshold)
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
        # general settings(train and inference)
        # model depth
        
        # augmentation count
        self.augment_count = QSpinBox()
        self.augment_count.setRange(1, 100)
        self.augment_count.setValue(5)
        self.augment_count.setSingleStep(1)
        # advanced(from preproc)
        self.horizontal_flip = QCheckBox("Horizontal Flip Augmentation")
        self.horizontal_flip.setChecked(True)
        self.brightness_aug = QCheckBox("Brightness Augmentation")
        self.brightness_aug.setChecked(True)
        self.rotation_aug = QCheckBox("Shift/Translation Augmentation")
        self.rotation_aug.setChecked(True)
        # brightness range controls
        self.brightness_min = QDoubleSpinBox()
        self.brightness_min.setRange(0.1, 2.0)
        self.brightness_min.setDecimals(2)
        self.brightness_min.setValue(0.8)
        self.brightness_min.setSingleStep(0.1)
        self.brightness_max = QDoubleSpinBox()
        self.brightness_max.setRange(0.1, 2.0)
        self.brightness_max.setDecimals(2)
        self.brightness_max.setValue(1.2)
        self.brightness_max.setSingleStep(0.1)
        # shift range control
        self.shift_max = QDoubleSpinBox()
        self.shift_max.setRange(0.01, 0.5)
        self.shift_max.setDecimals(2)
        self.shift_max.setValue(0.1)
        self.shift_max.setSingleStep(0.01)
        
        # below is inference settings
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
        
        self.back_btn3 = QPushButton("Back")
        layout.addWidget(QLabel("Augmentation Settings"))
        layout.addWidget(QLabel("Augment Count (per image)"))
        layout.addWidget(self.augment_count)
        layout.addWidget(self.horizontal_flip)
        layout.addWidget(self.brightness_aug)
        layout.addWidget(QLabel("Brightness Range"))
        layout.addWidget(self.brightness_min)
        layout.addWidget(self.brightness_max)
        layout.addWidget(self.rotation_aug)
        layout.addWidget(QLabel("Shift Max (fraction of size)"))
        layout.addWidget(self.shift_max)
        layout.addWidget(QLabel("Inference Settings"))
        layout.addWidget(QLabel("Side Len"))
        layout.addWidget(self.side_len)
        layout.addWidget(QLabel("Steps")) 
        layout.addWidget(self.steps)
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
        self.threshold.valueChanged.connect(lambda value: self.on_spin_changed("threshold", value)) # name, value
        self.side_len.editingFinished.connect(lambda: self.on_spin_changed("side_len", self.side_len.value()))
        self.steps.editingFinished.connect(lambda: self.on_spin_changed("steps", self.steps.value()))
        self.arch_depth.editingFinished.connect(lambda: self.on_spin_changed("arch_depth", self.arch_depth.value()))
        self.lr.valueChanged.connect(lambda value: self.on_spin_changed("lr", value))
        self.gen.editingFinished.connect(lambda: self.on_spin_changed("gen", self.gen.value()))
        self.augment_count.editingFinished.connect(lambda: self.on_spin_changed("augment_count", self.augment_count.value()))
        self.brightness_min.editingFinished.connect(lambda: self.on_spin_changed("brightness_min", self.brightness_min.value()))
        self.brightness_max.editingFinished.connect(lambda: self.on_spin_changed("brightness_max", self.brightness_max.value()))
        self.shift_max.editingFinished.connect(lambda: self.on_spin_changed("shift_max", self.shift_max.value()))
        self.horizontal_flip.stateChanged.connect(lambda state: self.on_checkbox_changed("flip_enabled", state))
        self.brightness_aug.stateChanged.connect(lambda state: self.on_checkbox_changed("brightness_enabled", state))
        self.rotation_aug.stateChanged.connect(lambda state: self.on_checkbox_changed("shift_enabled", state))
        
    
    def update_frame(self, frame):
        import cv2
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
    
    # multi-class if either checkbox is checked
    @property
    def multi_class_enabled(self):
        return (self.m_class_cbox1.isChecked() or self.m_class_cbox2.isChecked())
    
    def run(self):
        from ui.worker import MLWorker
        if self.worker and self.worker.isRunning():
            return
        
        update_settings(self.get_settings())
        
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
            arch_depth=self.arch_depth.value(),
            label_widget=self.video_label,
        )
        target_log = self.log1 if selected_mode in train_modes else self.log2
        target_progress = self.progress1 if selected_mode in train_modes else None
        
        self.worker.frame.connect(self.update_frame)
        self.worker.log.connect(lambda s: target_log.append(str(s)))
        self.worker.progress.connect(target_progress.setValue) if target_progress else None
        self.worker.finished.connect(self.on_done)
        self.worker.start()
    
    # train window part
    def on_radio_toggled(self, index, checked):
        if checked:
            print(f"Radio button {index+1} picked")
    
    def on_checkbox_changed(self, param_name, state):
        value = state == 2  # 2 for checked, 0 for unchecked
        self.settings[param_name] = value
        update_settings({param_name: value})
        print(f"{param_name}: {value}")
              
    # settings logic part
    def on_spin_changed(self, name, value):
        self.settings[name] = value
        update_settings({name: value})
        print(self.settings)

    # get all settings at once for convenience
    def get_settings(self):
        return {
            "threshold": self.threshold.value(),
            "side_len": self.side_len.value(),
            "steps": self.steps.value(),
            "arch_depth": self.arch_depth.value(),
            "lr": self.lr.value(),
            "gen": self.gen.value(),
            "augment_count": self.augment_count.value(),
            "flip_enabled": self.horizontal_flip.isChecked(),
            "brightness_enabled": self.brightness_aug.isChecked(),
            "brightness_min": self.brightness_min.value(),
            "brightness_max": self.brightness_max.value(),
            "shift_enabled": self.rotation_aug.isChecked(),
            "shift_max": self.shift_max.value(),
        }
    
    # the rest of general logic
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