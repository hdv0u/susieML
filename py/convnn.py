# susie cnn v1.01
# fixed confidence output issues
# plan is to add app-like support

import torch
import torch.nn as nn
# ui imports
from ui.file_dialog import save_model_file, select_model_file, labeled_picker

# the brain; input = 128x128x3
class SussyCNN(nn.Module):
    def __init__(self, input_size:int=128, out_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4)),
        )
        # hotfix
        with torch.no_grad():
            dummy = torch.zeros(1,3,input_size, input_size)
            flat_size = self.features(dummy).view(1,-1).size(1)
            
        self.out_channels = out_channels
        self.classifier = nn.LazyLinear(self.out_channels)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


def main(
    mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None, parent=None,
    model_save=None, train_paths=None, train_labels=None, load_model=None, multi_class=False,
    label_widget=None
    ):
    import os
    import config
    # core imports
    from core.model.convnn_runner import CNNTrainer, CNNInference
    from core.frame_sources import screen_source
    from preproc import new_augment
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config.CNN
    torch.backends.cudnn.benchmark = True
    log_fn(f"using {device}")
    
    model_constructor = SussyCNN
    
    if mode == "3":
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
        trainer = CNNTrainer(
            model_fn=lambda: model_constructor(out_channels=num_classes),
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
    
    elif mode == "4":
        if load_model is None:
            load_model = select_model_file(parent=parent)
            if not load_model:
                log_fn("inference canceled (no model)")
                return
            
        if not os.path.exists(load_model):
            log_fn(f"model FNF: {load_model}")
            return
        
        from core.registry import validate_model_file
        out_classes = validate_model_file(load_model, expected_mode=4, log_fn=log_fn)
        if out_classes is None:
            log_fn("Checkpoint validation failed")
            return
        
        model = SussyCNN(out_channels=out_classes).to(device)
        
        try:
            saved_state = torch.load(load_model, map_location=device)
            model.load_state_dict(saved_state)
        except RuntimeError as e:
            log_fn(f"Failed to load checkpoint into convnn: {e}")
            return
        
        model.eval()
        backend = CNNInference(model, device, cfg, log_fn=log_fn, multi_class=(out_classes>1), num_classes=out_classes)
        
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