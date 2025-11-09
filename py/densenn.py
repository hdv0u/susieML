# susieML v1 NumPy edition(very inefficient)
# target goal for today:
# 300x2 train image with learn = 0.01-0.05 and neuron set = 1024
import cv2, time, os, mss
import numpy as np
from shared import outputImgP
from preproc import image_proc, new_augment
from imagePicker import save_model, load_model

class denseSussyML:
    def __init__(self, input_size, hidden_size=512, output_size=1):
        self.weightbias1 = np.random.randn(input_size, hidden_size) * 0.01
        self.biasdense1 = np.zeros((hidden_size,))
        self.weightdense2 = np.random.randn(hidden_size, output_size) * 0.01
        self.biasdense2 = np.zeros((output_size,))
        self.epsilon = 1e-9
    def forward(self, X):
        self.hidden = np.maximum(0, X @ self.weightbias1 + self.biasdense1) # relu deriv?
        z = self.hidden @ self.weightdense2 + self.biasdense2
        return 1 / (1 + np.exp(-z))
    def backward(self, X, y, learn=0.01):
        pred = self.forward(X)
        error = pred - y
        grad_weight2 = self.hidden.T @ error / X.shape[0]
        grad_bias2 = np.mean(error, axis=0)
        d_hidden = (error @ self.weightdense2.T) * (self.hidden > 0)
        grad_weight1 = X.T @ d_hidden / X.shape[0]
        grad_bias1 = np.mean(d_hidden, axis=0)
        
        self.weightdense2 -= learn * grad_weight2
        self.biasdense2 -= learn * grad_bias2
        self.weightbias1 -= learn * grad_weight1
        self.biasdense1 -= learn * grad_bias1
        # ahhh BCE cuz if susie yes or no ahhh
        loss = -np.mean(y * np.log(pred + self.epsilon) + (1-y) * np.log(1-pred + self.epsilon))
        return loss
    def train(self, X, y, generations=100, denseLearn=0.01, X_test=None, y_test=None):
        for gen in range(generations):
            loss = self.backward(X, y, learn=denseLearn)
            if gen % 1 == 0:
                if X_test is not None and y_test is not None:
                    pred_test = self.forward(X_test)
                    # test BCE incase im lost
                    test_loss = -np.mean(y_test * np.log(pred_test + self.epsilon) +
                                         (1-y_test) * np.log(1-pred_test + self.epsilon))
                    print(f"gen:{gen}, train loss:{loss:.5f}, test loss:{test_loss:.5f}")
                else:
                    print(f"gen:{gen}, train loss:{loss:.5f}")
    def predict(self, X):
        return self.forward(X)
    
    def save_weights(self, path="nn_weights.npz"):
        np.savez(path, w1=self.weightbias1, b1=self.biasdense1, w2=self.weightdense2, b2=self.biasdense2)
    def load_weights(self, path="nn_weights.npz"):
        data = np.load(path)
        self.weightbias1 = data['w1']
        self.biasdense1 = data['b1']
        self.weightdense2 = data['w2']
        self.biasdense2 = data['b2']

# input moved out to main
def main(mode):
    
    if mode == "1":
        saved_model = save_model()
        # pick file once
        train_path = outputImgP()
        if isinstance(train_path, tuple):
            train_path = train_path[0]
        if len(train_path) == 0:
            print("No files selected. susie out")
            exit()
        # assign label on file/folder
        train_labels = [[1] if 'pos' in p.lower() else [0] for p in train_path]
        print("Training paths:", train_path.flatten())
        print("Training labels:", train_labels)
        
        train_input, checker_train = new_augment(train_path, train_labels, augment_count=10, mode='dense')
        # test input
        test_path = []
        test_labels = [[1]]
        test_input = np.array([image_proc(p) for p in test_path]).reshape(len(test_path), -1)
        checker_test = np.array(test_labels).reshape(len(test_path), 1)
        # start train
        nn = denseSussyML(input_size=train_input.shape[1])
        nn.train(train_input, checker_train, generations= 100, X_test=test_input, y_test=checker_test)
        # save weights for detection
        nn.save_weights(saved_model)

    elif mode == "2":
        loaded_model = load_model()
        modelPath = loaded_model
        input_size = 128*128*3  # adjusted to preproc
        cooldown = 0.5
        threshold = 0.3 # 0.67 as default
        channels = 3
        # load model
        nn = denseSussyML(input_size=input_size)
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"model file not found: {modelPath}")
        nn.load_weights(modelPath)
        print("weights loaded well!")
        # screenrec thing
        sct = mss.mss()
        monitor = sct.monitors[1]
        lastDetectionTime = 0
        sideLen = int((input_size / channels) ** 0.5)
        while True:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            frame_small = cv2.resize(frame, (sideLen, sideLen)) # match training and norms
            
            X = frame_small.flatten().reshape(1, -1) / 255.0
            if X.shape[1] != nn.weightbias1.shape[0]:
                raise ValueError(f"input {X.shape[1]} didnt match the network {nn.weightbias1.shape[0]}")
            # prediction time
            pred = nn.predict(X)[0][0]
            current_time = time.time()
            
            if pred >= threshold and current_time - lastDetectionTime > cooldown:
                lastDetectionTime = current_time
                print(f"sussy maybe found..? confidence lvl: {pred:.2f}")
                # some visual feedback
                cv2.putText(frame, f"Detected! {pred:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
                time.sleep(0.5)
                
            # visual Output
            cv2.imshow("bleh twan detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cv2.destroyAllWindows()
    else: print("hell naw..")