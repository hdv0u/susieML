# susieML v2
# target goal for today:
# 300x2 train image with learn = 0.01-0.05 and neuron set = 1024
import torch, cv2, time, mss, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ui.file_dialog import save_model_file, select_model_file, labeled_picker
from preproc import new_augment, image_proc
from core.model.convnn_runner import CNNTrainer, CNNInference
from core.frame_sources import screen_source
import config

class DenseSussy(nn.Module):
    def __init__(self, input_size=128*128*3, hidden_size=512, num_classes=1) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def main(
    mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None,
    parent=None, model_save=None, train_paths=None, train_labels=None,
    load_model=None, multi_class=False, label_widget=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = config.DENSE
    log_fn(f"using {device}")
    
    if mode == '1':
        if model_save is None:
            model_save = save_model_file(parent=parent)
            if not model_save:
                log_fn("train canceled (no save path)")
                return
            
        if train_paths is None:
            train_paths, train_labels = labeled_picker(parent=parent)
            if not train_paths:
                log_fn("train canceled (no images)")
                return
        
        num_classes = 3 if multi_class else 1
        model_constructor = lambda: DenseSussy(input_size=cfg["input_size"], hidden_size=512, num_classes=num_classes)
        trainer = CNNTrainer(
            model_fn=model_constructor,
            device=device,
            cfg=cfg,
            log_fn=log_fn,
            progress_fn=progress_fn,
            stop_fn=stop_fn,
            multi_class=multi_class
        )
        trainer.train_from_paths(
            train_paths=train_paths,
            save_path=model_save,
            augment_fn=new_augment,
            labels=train_labels
        )
        return
    
    elif mode == '2':
        if load_model is None:
            load_model = select_model_file(parent=parent)
            if not load_model or not os.path.exists(load_model):
                log_fn("inference canceled (no model)")
                return
            
        if not os.path.exists(load_model):
            log_fn(f"model FNF: {load_model}")
            return
        
        saved_state = torch.load(load_model, map_location=device)
        fc_key = next((k for k in saved_state if 'fc2.weight' in k), None)
        if fc_key is None:
            log_fn("Invalid DenseNN checkpoint")
            return
        output_size = saved_state[fc_key].shape[0]
        log_fn(f"detected output size from model: {output_size}")
        
        model = DenseSussy(input_size=cfg['input_size'], hidden_size=512, num_classes=output_size).to(device)
        model.load_state_dict(saved_state)
        model.eval()
        backend = CNNInference(model, device, cfg, log_fn=log_fn, multi_class=(output_size > 1), num_classes=output_size)
        
        from core.inference_runner import run_inference
        run_inference(
            frame_source=screen_source,
            frame_sink=frame_fn,
            backend=backend,
            stop_fn=stop_fn,
            log_fn=log_fn
        )
        return
    else:
        log_fn("Invalid mode..")
        return
    