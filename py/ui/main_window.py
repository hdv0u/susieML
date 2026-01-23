
# debug: print(f"debug: widget {__name__}")

class TestUI:
    def __init__(self):
        from PyQt5.QtWidgets import (
            QWidget, QLabel, QRadioButton, QPushButton,
            QVBoxLayout, QButtonGroup, QTextEdit, QProgressBar, QCheckBox
        )
        from ui.worker import MLWorker
        from core.registry import MODELS
        from ui.file_dialog import save_model_file, select_model_file, labeled_picker, labeled_picker_multi
        
        class _W(QWidget):
            pass
        self._widget = _W()
        self._widget.setWindowTitle("Test UI")
        
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
        self.log.append(text)
    
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
        
        if selected_mode == '3':
            model_save = self._save_model_file(parent=self._widget)
            if not model_save:
                self.log.append("no save path selected")
                self._reset_ui()
                return
            if self._multi_class_enabled:
                multi_class = True
                class_names = ["Positive", "Neutral", "Negative"]
            else:
                multi_class = False
                class_names = ["Positive", "Negative"]
            
            class_map = {name: idx for idx, name in enumerate(class_names)}
            train_paths, labels = self._labeled_picker_multi(class_names=class_names)
            if not train_paths:
                self.log.append("no training images selected")
                self._reset_ui()
                return
            
        elif selected_mode == '4':
            load_model = self._select_model_file(parent=self._widget)
            if not load_model:
                self.log.append("no model selected")
                self._reset_ui()
                return
        
        self.worker = self._MLWorker(
            mode=selected_mode,
            model_save=model_save,
            train_paths=train_paths,
            train_labels=labels if selected_mode=='3' else None,
            load_model=load_model,
            multi_class=self._multi_class_enabled,
            label_widget=self.video_label,
        )
        self.worker.log.connect(lambda s: self.log.append(str(s)))
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_done)
        self.worker.start()
        
    def stop(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("stopping..")
              
    def on_done(self):
        self.status.setText("done")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        
    def _reset_ui(self):
        self.status.setText("idle")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
            