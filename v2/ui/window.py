from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QTextEdit, QStackedWidget,
    QRadioButton, QSpinBox, QCheckBox,
    QProgressBar, QButtonGroup
)
from PyQt5.QtWidgets import QFileDialog
from ui.mediator_bus import bus
from ui.state_tracker import StateTracker
import json

class TestWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("SusieML")
        self.setMinimumSize(854, 480)

        self.layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        self.worker = None
        self.settings = {}
        self._last_result = None
        self.dataset_path = None
        self.state_tracker = StateTracker()

        self.menu_window()
        self.train_window()
        self.run_window()
        self.settings_window()
        self.settings_logic()
        self.state_tracker.diff(self)

        self.stack.setCurrentIndex(0)
        
    def menu_window(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("MAIN MENU"))

        train_btn = QPushButton("Train")
        run_btn = QPushButton("Run")
        settings_btn = QPushButton("Settings")

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

        self.gen = QSpinBox()
        self.gen.setRange(1, 10000)
        self.gen.setValue(50)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(1e-6, 1e-1)
        self.lr.setDecimals(6)
        self.lr.setValue(5e-4)

        self.m_class_cbox1 = QCheckBox("Enable Multi-class")

        self.log1 = QTextEdit()
        self.log1.setReadOnly(True)

        self.progress1 = QProgressBar()

        self.radio_btns1 = []
        self.train_mode_group = QButtonGroup(self)

        layout.addWidget(QLabel("Train"))

        # placeholder modes (no external MODELS dependency)
        for i in range(3):
            rb = QRadioButton(f"Mode {i}")
            self.train_mode_group.addButton(rb, i)
            layout.addWidget(rb)
            self.radio_btns1.append(rb)
            
        self.quick_train_btn = QPushButton("Quick Train")
        self.train_btn = QPushButton("Train")
        self.stop_btn1 = QPushButton("Stop")
        self.back_btn1 = QPushButton("Back")

        layout.addWidget(self.m_class_cbox1)
        layout.addWidget(self.log1)
        layout.addWidget(self.progress1)

        layout.addWidget(QLabel("Architecture Depth"))
        layout.addWidget(self.arch_depth)

        layout.addWidget(QLabel("Epochs"))
        layout.addWidget(self.gen)

        layout.addWidget(QLabel("Learning Rate"))
        layout.addWidget(self.lr)

        layout.addWidget(self.quick_train_btn)
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
        self.video_label.setMaximumSize(640, 360)

        self.m_class_cbox2 = QCheckBox("Multi-class")

        self.run_btn = QPushButton("Run")
        self.stop_btn2 = QPushButton("Stop")

        self.log2 = QTextEdit()
        self.log2.setReadOnly(True)
        self.log2.setMaximumHeight(100)

        self.back_btn2 = QPushButton("Back")

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setValue(0.67)

        self.mode_group = QButtonGroup(self)
        self.radio_btns2 = []

        controls = QVBoxLayout()

        # placeholder modes (no external MODELS dependency)
        for i in range(3):
            rb = QRadioButton(f"Mode {i}")
            self.mode_group.addButton(rb, i)
            controls.addWidget(rb)
            self.radio_btns2.append(rb)

        controls.addWidget(self.m_class_cbox2)
        controls.addWidget(QLabel("Threshold"))
        controls.addWidget(self.threshold)

        controls.addWidget(self.log2)
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

        self.augment_count = QSpinBox()
        self.augment_count.setRange(1, 100)
        self.augment_count.setValue(5)

        self.horizontal_flip = QCheckBox("Horizontal Flip")
        self.brightness_aug = QCheckBox("Brightness")
        self.rotation_aug = QCheckBox("Shift")

        self.brightness_min = QDoubleSpinBox()
        self.brightness_min.setValue(0.8)

        self.brightness_max = QDoubleSpinBox()
        self.brightness_max.setValue(1.2)

        self.shift_max = QDoubleSpinBox()
        self.shift_max.setValue(0.1)

        self.side_len = QSpinBox()
        self.side_len.setRange(1, 512)
        self.side_len.setValue(128)

        self.steps = QSpinBox()
        self.steps.setRange(1, 512)
        self.steps.setValue(96)

        self.back_btn3 = QPushButton("Back")

        layout.addWidget(QLabel("Augmentation Settings"))
        layout.addWidget(self.augment_count)

        layout.addWidget(self.horizontal_flip)
        layout.addWidget(self.brightness_aug)
        layout.addWidget(self.rotation_aug)

        layout.addWidget(QLabel("Brightness Min"))
        layout.addWidget(self.brightness_min)

        layout.addWidget(QLabel("Brightness Max"))
        layout.addWidget(self.brightness_max)

        layout.addWidget(QLabel("Shift Max"))
        layout.addWidget(self.shift_max)

        layout.addWidget(QLabel("Inference Settings"))
        layout.addWidget(QLabel("Side Length"))
        layout.addWidget(self.side_len)

        layout.addWidget(QLabel("Steps"))
        layout.addWidget(self.steps)

        layout.addWidget(self.back_btn3)

        self.stack.addWidget(page)

    def settings_logic(self):
        self.back_btn1.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.back_btn2.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.back_btn3.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        # connect value changes to state tracker for logging
        self.gen.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.lr.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.arch_depth.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.side_len.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.steps.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.threshold.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.augment_count.valueChanged.connect(lambda: self.state_tracker.diff(self))
        self.quick_train_btn.clicked.connect(self.quick_train)
        self.train_btn.clicked.connect(self.on_train)
        self.run_btn.clicked.connect(self.run)

        self.stop_btn1.clicked.connect(self.stop)
        self.stop_btn2.clicked.connect(self.stop)
    
    # makes JSON from pos and neg, saves and returns path, for quick train
    def json_gen(self):
        pos, _ = QFileDialog.getOpenFileNames(
            self,
            "Positive Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not pos:
            return None
        
        neg, _ = QFileDialog.getOpenFileNames(
            self,
            "Negative Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not neg:
            return None
        
        dataset = {
            "version": 2,
            "pos": pos,
            "neg": neg,
            "labels": {
                "pos": 1,
                "neg": 0
            }
        }
        
        from core.paths import ROOT
        from pathlib import Path
        
        dataset_dir = ROOT / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = dataset_dir / "dataset.json"
                
        if not save_path:
            return None
        
        # debug shi
        print(f"pos: {pos}")
        print(f"neg: {neg}")
        print(f"save_path: {save_path}")
        
        save_path = Path(save_path)
        
        if save_path.suffix != ".json":
            save_path = save_path.with_suffix(".json")
            
        with open(save_path, "w") as f:
            json.dump(dataset, f, indent=4)
        
        self.dataset_path = save_path
        
        return str(save_path)
        
    def quick_train(self):
        from ui.mediator_bus import bus

        json_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset JSON",
            "",
            "JSON (*.json)"
        )
        
        if not json_path:
            self.log1.append("no json found")
            return
        
        bus.train_requested.emit({
            "dataset_path": json_path,
            "config": self.build_config()
        })
        
        self.log1.append(f"quick train started! {json_path}")
    
    def run(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "Model (*.pt *.pth)"
        )
        
        if model_path:
            bus.run_requested.emit(model_path)

    def on_train(self):
        json_path = self.json_gen()
        
        if not json_path:
            self.log1.append("dataset creation failed")
            return
        
        self.log1.append(json.dumps(self.build_config(), indent=4))
            
        bus.train_requested.emit({
            "dataset_path": json_path,
            "config": self.build_config()
        })
    
    # build config dict from UI settings
    def build_config(self):
        return {
            "training": {
                "epochs": self.gen.value(),
                "learning_rate": self.lr.value(),
                "arch_depth": self.arch_depth.value(),
                "batch_size": 16,
            },
            "inference": {
                "side_len": self.side_len.value(),
                "steps": self.steps.value(),
                "threshold": self.threshold.value(),
            },
            "augment": {
                "augment_count": self.augment_count.value(),
                "flip": self.horizontal_flip.isChecked(),
                "brightness": self.brightness_aug.isChecked(),
                "shift": self.rotation_aug.isChecked(),
            },
            "model": {
                "multi_class": self.m_class_cbox1.isChecked(),
                "out_channels": 1 if not self.m_class_cbox1.isChecked() else 2,
                "input_size": (self.side_len.value(), self.side_len.value()),
                
            }
        }
    
    # didnt fully stop inference, still WIP
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        
        self.log1.append("Train stopped")
        self.log2.append("Inference stopped")

    def set_mediator(self, mediator):
        self.mediator = mediator
    
    # update frame with results as logs n display it on UI
    def update_frame(self, data):
        import cv2
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import Qt
        frame, result = data
        self._last_frame = frame
        self._last_result = result

        text = ""
        
        # confidence display logic
        if isinstance(result, dict):
            conf = result.get("confidence")
            if conf is not None:
                text = f"{conf:.2f}"
        elif isinstance(result, (int, float)):
            text = f"{result:.2f}"
            
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w # calculate bytes/line based on channels and width
        
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_result(self, result):
        self._last_result = result
        self.log2.append(str(result))