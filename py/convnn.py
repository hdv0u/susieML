# susie cnn v1.2
# c1 can do 200-1,000 images, c2 can do 1,000-5,000, c3 can do 5,000-15,000, c4 can do 15,000-50,000, c5 can do 50,000-200,000

import torch
import torch.nn as nn

# the brain; input = 128x128x3
class SussyCNN(nn.Module):
    def __init__(self, input_size:int=128, out_channels=1, depth=1):
        super().__init__()
        mult = max(1, depth)
        c1 = 16 * mult
        c2 = 32 * mult
        c3 = 64 * mult
        c4 = 128 * mult
        c5 = 256 * mult
        self.features = nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1,c1,3,padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(c1,c2,3,padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2,c2,3,padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(c2,c3,3,padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3,c3,3,padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(c3,c4,3,padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),

            nn.Conv2d(c4,c4,3,padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(c4,c5,3,padding=1),
            nn.BatchNorm2d(c5),
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

def _normalize_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def infer_convnn_depth_from_state(state_dict, log_fn=print):
    state_dict = _normalize_state_dict(state_dict)
    if "features.0.weight" not in state_dict:
        log_fn("Unable to infer ConvNN depth from checkpoint")
        return None
    channels = state_dict["features.0.weight"].shape[0]
    if channels % 16 != 0:
        log_fn(f"Unexpected ConvNN first layer channels: {channels}")
        return None
    return max(1, min(8, channels // 16))

def main(
    mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None, parent=None,
    model_save=None, train_paths=None, train_labels=None, load_model=None, multi_class=False,
    label_widget=None, arch_depth=None
    ):
    import os
    import config
    # core imports
    from core.model.convnn_runner import CNNTrainer, CNNInference
    from core.frame_sources import screen_source
    from preproc import new_augment
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = config.CNN
    depth = max(1, min(8, arch_depth or cfg.get('arch_depth', 1)))
    torch.backends.cudnn.benchmark = True
    log_fn(f"using {device}")
    
    model_constructor = SussyCNN
    
    if mode == "3":
        from ui.file_dialog import save_model_file, labeled_picker

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
            model_fn=lambda: model_constructor(out_channels=num_classes, depth=depth),
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
        from ui.file_dialog import select_model_file

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
        
        saved_state = torch.load(load_model, map_location=device)
        saved_depth = infer_convnn_depth_from_state(saved_state, log_fn=log_fn)
        if saved_depth is None:
            log_fn("Failed to infer ConvNN depth from checkpoint")
            return
        if saved_depth != depth:
            log_fn(f"Using checkpoint depth {saved_depth} instead of requested {depth}")
        model = SussyCNN(out_channels=out_classes, depth=saved_depth).to(device)
        
        try:
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