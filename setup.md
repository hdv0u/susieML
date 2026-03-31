# **Requirements:**

- Python 3.13.x
- a terminal
- install requirements.txt
- Your dataset(optional)


## **How to run:**
1. Go to CMD or to any terminal
2. Go to the folder with *"cd SusieML\py"* in the terminal
3. Install the requirements with *"pip install -r requirements.txt"* in the terminal
4. After installing the req, run main.py with either *"py main.py"* or *"python main.py"*
   - 4.1. If either one fails to run the GUI, run the main with the "--cli" tag after it, if it still fails, you must check your current python version and must match from the requirements.

## **How to use:**

**A. Train**
1. Pick the desired model architecture
   - 1.a. Dense-type model is usually good for MNIST-type datasets
   - 1.b. CNN-type model is recommended for actual image detection
   - 1.c. Resnet-type model is recommended for deeper detection
   - 1.1. Depth can go from 1-8 for the CNN architecture(implementing custom depth for all arch soon), c1 can do 200-1,000 images, c2 can do 1,000-5,000, c3 can do 5,000-15,000, c4 can do 15,000-50,000, c5 can do 50,000-200,000 images
2. Pick a save directory(default is in models folder)
3. Train your model with the chosen model and dataset, should be organized for convenience
4. Wait and watch!

**B. Inference/Detect**
1. Pick inference model architecture
2. Pick a loaded model from the trained model or external one
3. Enjoy detecting Susie(or whatever based on your imageset u fed on)
