import torch, os
import torch.nn as nn
from core.model.convnn_runner import CNNTrainer, CNNInference
from core.frame_sources import screen_source
from ui.file_dialog import save_model_file, select_model_file, labeled_picker
from preproc import new_augment
import config
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 
               
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class SussyResNet(nn.Module):
    def __init__(self, layers=[3,4,6,3], num_class=1) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1],stride=2)
        self.layer3 = self._make_layer(256, layers[2],stride=2)
        self.layer4 = self._make_layer(512, layers[3],stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Bottleneck.expansion, num_class)
        
    def _make_layer(self, mid_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels*Bottleneck.expansion,1,stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels*Bottleneck.expansion)
            )
        
        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = mid_channels * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def main(
    mode, log_fn=print, frame_fn=None, progress_fn=None, stop_fn=None,
    parent=None, model_save=None, train_paths=None, train_labels=None,
    load_model=None, multi_class=False, label_widget=None
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = config.RESNET
    log_fn(f"using {device}")
    
    if mode == '5':
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
        model_constructor = lambda: SussyResNet(num_class=num_classes) 
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
    
    elif mode == '6':
        if load_model is None:
            load_model = select_model_file(parent=parent)
            if not load_model or not os.path.exists(load_model):
                log_fn("inference canceled (no model)")
                return
        
        saved_state = torch.load(load_model, map_location=device)
        fc_key = next((k for k in saved_state if 'fc.weight' in k), None)
        if fc_key is None:
            log_fn("Invalid ResNet checkpoint")
            return
        output_size = saved_state['fc.weight'].shape[0]
        log_fn(f"detected output size from model: {output_size}")
        
        model = SussyResNet(num_class=output_size).to(device)
        model.load_state_dict(torch.load(load_model, map_location=device))
        model.eval()
        backend = CNNInference(model, device, cfg, log_fn=log_fn,multi_class=(output_size>1), num_classes=output_size)
        
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
        log_fn("invalid mode")
        return