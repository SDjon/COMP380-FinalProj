# COMP380-FinalProj
[**Authors**] Jonathan Rivera, Gianpaolo Tabora, Kian Drees, Precee-Noel Ginigeme <br />
[**Class**] COMP 380, Neural Networks <br />
[**Instructor**] Professor Eric Jiang <br />
[**Title**] LeafLens AI: An Image Classification Network for Automated Pest Control <br />

[**Repo layout**] 
- train Folder: Contains 13 different insects.
- Testing_data Folder: Contains same training images as to test the classification.
- small_test: Small test for the spary or no spray function of LeafLens.
- best_model.h5: The model of LeafLens we are using.
- class_indices.json: Indecies associated with insects.
- LeafLens.py: Main working file of LeafLens.
- test.csv: File to access images for LeafLens for testing
- train.csv: File to access images for LeafLens for training
- oldVersions Folder: Contains outdated versions. So show out progress.
  1. BackPropogationNet.java: Past working file where we built the neural net from scratch.
  2. PestClassifer (old, k-fold).py: Old LeafLens working file that worked with k-fold.
  3. PestClassifer (old).py: Old LeafLens working file.
  4. WeightContainer.java: Old helper file for BackPropogationNet.java.



[**Required Packages**]
- TensorFlow → neural network framework
- NumPy → numerical operations and arrays
- Pillow → image loading and processing
- scikit-learn → train/test split and machine learning tools
- pandas → reads csv files
- pyqt5 → GUI for visualization and live testing
