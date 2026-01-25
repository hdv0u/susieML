from convnn import SussyCNN
from densenn import DenseSussy
from resnet1 import SussyResNet
# buttons registry
# this is where u put them buttons in the UI
MODELS = {
    "1": {
        "module": "models.densenn",
        "label": "Train (dense)",
    },
    "2": {
        "module": "models.densenn",
        "label": "Detect (dense)",
    },
    "3": {
        "module": "models.convnn",
        "label": "Train (CNN)",
    },
    "4": {
        "module": "models.convnn",
        "label": "Detect (CNN)",
    },
    "5": {
        "module": "models.resnet1",
        "label": "Train (ResNet50)",
    },
    "6": {
        "module": "models.resnet1",
        "label": "Detect (ResNet50)",
    },
}

def get_model_constructor(mode_id):
    return MODELS.get(str(mode_id), {}).get("constructor", None)

def detect_arch_from_state(state_dict):
    # dense
    if any('fc2.weight' in k for k in state_dict):
        return 'models.densenn'
    
    # resnet
    if 'conv1.weight' in state_dict and any(k.startswith('layer1') for k in state_dict):
        return 'models.resnet1'
    
    # convnn
    convnn_keys = ['features.0.weight', 'conv_layers.0.weight', 'conv_out.weight', 'classifier.weight']
    if any(k in state_dict for k in convnn_keys):
        return 'models.convnn'
    
    return None

import torch, os
def validate_model_file(load_model, expected_mode, log_fn=print):
    if not load_model or not os.path.exists(load_model):
        log_fn(f"model FNF: {load_model}")
        return None
    
    try:
        state_dict = torch.load(load_model, map_location='cpu')
    except Exception as e:
        log_fn(f"Failed to read model file: {e}")
        return None
    
    module_name = MODELS[str(expected_mode)]['module']
    arch_detected = detect_arch_from_state(state_dict)
    
    if arch_detected is None:
        log_fn("unable to detect architecture from checkpoint keys")
        return None
    
    if arch_detected != module_name:
        log_fn(f"Architecture mismatch: expected {module_name}, found {arch_detected}")
        return None
    
    
    out_classes = None
    for key in state_dict.keys():
        if 'fc.weight' in key or 'classifier.weight' in key or 'fc2.weight' in key:
            out_classes = state_dict[key].shape[0]
            break
        
    if out_classes is None:
        log_fn("Cannot detect output classes from checkpoint")
        return None
    
    log_fn(f"Model validated: {arch_detected}, output_classes={out_classes}")
    return out_classes