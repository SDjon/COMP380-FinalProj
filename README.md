# COMP380-FinalProj
[**Authors**] Jonathan Rivera, Gianpaolo Tabora, Kian Drees, Precee-Noel Ginigeme <br />
[**Class**] COMP 380, Neural Networks <br />
[**Instructor**] Professor Eric Jiang <br />
[**Title**] LeafLens AI: An Image Classification Network for Automated Pest Control <br />

[**Repo layout**] 
- Training Folder: Contains 4 different pests with 4 different angles for training. Grasshopper, Snail, Rat, & Rabbit
- Testing Folder: Contains 3 different folders with no/low/medium corruption testing images
  1. Control_test Folder: Contains control images to see if the neural network
  2. Low_test Folder: Contains images with 10% corruption
  3. Med_test Folder: Contains images with 20% corruption
- BackPropogationNet.java: Main working file.

[**Required Packages**]
- TensorFlow → neural network framework
- NumPy → numerical operations and arrays
- Pillow → image loading and processing
- matplotlib → data visualization and plotting
- scikit-learn → train/test split and machine learning tools
You can install all required packages at once with: 
- pip install tensorflow numpy pillow matplotlib scikit-learn
