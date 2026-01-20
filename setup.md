# **Requirements:**

- Python 3.13.x
- a terminal
- install requirements.txt
- Your dataset(optional)


## **How to run:**
1. Go to CMD or to any terminal
2. Go to the folder with *"cd SusieML\py"* in the terminal
3. Run main.py with either *"py main.py"* or *"python main.py"*
   - 3.1. If either one fails to run the GUI, check current python version and must match from the requirements.

## **How to use:**

**A. Train**
1. Pick the desired model architecture
   - 1.a. Dense-type model is usually good for MNIST-type datasets
   - 1.b. CNN-type model is recommended for actual image detection
   - 1.c. Resnet-type model is recommended for deeper detection
2. Pick a save directory(default is in models folder)
3. Train your model with the chosen model and dataset, should be organized for convenience
4. Wait and watch it train and finish

**B. Inference/Detect**
1. Pick inference model architecture(future update should unify it into one button)
2. Pick a loaded model from the trained model or external one
3. Check results
