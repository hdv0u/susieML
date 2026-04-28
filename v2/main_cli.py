import argparse
from config import cfg
from core.control.config_controller import ConfigController
from core.control.run_controller import RunController

from core.usecases.train import run_train
from core.usecases.infer import build_detector

def main():
    parser = argparse.ArgumentParser(description="SusieML Command Line Interface", prog="susieml")
    
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", required=True, help="Path to training dataset")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--lr", type=float)
    
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model", required=True, help="Path to trained model")
    infer_parser.add_argument("--threshold", type=float)
    infer_parser.add_argument("--side", type=int)
    infer_parser.add_argument("--steps", type=int)
    
    args = parser.parse_args()
    
    cfg_ctrl = ConfigController(cfg)
    stop_ctrl = RunController()
    
    if args.command == "train":
        if args.epochs:
            cfg_ctrl.set_value("training", "epochs", args.epochs)
            
        if args.lr:
            cfg_ctrl.set_value("training", "learning_rate", args.lr)
            
        model_path = run_train(
            cfg_ctrl=cfg_ctrl,
            dataset_path=args.dataset,
            log_fn=print,
            progress_fn=lambda p: print(f"Progress: {p:.2f}%"),
            stop_ctrl=stop_ctrl
        )
        
        print(f"train done, me stop now. Model saved at: {model_path}")
        
    elif args.command == "infer":
        if args.threshold:
            cfg_ctrl.set_value("inference", "threshold", args.threshold)
            
        if args.side:
            cfg_ctrl.set_value("inference", "side_length", args.side)
            
        if args.steps:
            cfg_ctrl.set_value("inference", "steps_per_side", args.steps)
        
        detector = build_detector(
            cfg_ctrl=cfg_ctrl,
            model_path=args.model,
            log_fn=print
        )
        
        print(f"infer started. Model loaded from: {args.model}")
        
        import cv2
        from mss import mss
        import numpy as np
        
        sct = mss()
        monitor = sct.monitors[1]
        
        try:
            while True:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                frame, result = detector.detect(frame)
                
                print(f"Inference result: {result}")
                
                cv2.imshow("infer", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nStopped inference.")
    
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()
    