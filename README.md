# CNN Classification – Cats vs Dogs 🐱🐶

## 🧠 Overview
This project implements a *Convolutional Neural Network (CNN)* for *image classification* using TensorFlow & Keras.  
The dataset consists of *4000 training images* and *1000 testing images* of *cats and dogs*.  
The model is trained to classify an input image as either *Cat* or *Dog*.  

## 📖 About CNN
A *Convolutional Neural Network (CNN)* is designed for image-related tasks.  
It uses *Convolution layers (Conv2D)* to capture spatial features, *Pooling layers (MaxPooling2D)* to reduce dimensions, and *Dense layers* to perform classification.  
The *sigmoid activation* function in the output layer is used for *binary classification*.  

## 🛠 Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib  
- Scikit-learn  
- Keras Preprocessing (ImageDataGenerator, image preprocessing utilities)  

## ⚙️ Workflow
1. *Dataset Preparation*  
   - Used ImageDataGenerator from Keras preprocessing to load images directly from folders.  
   - Applied *rescaling* and basic preprocessing.  

2. *Model Creation*  
   - Built CNN using *Sequential API* with layers:  
     - Conv2D (for feature extraction)  
     - MaxPooling2D (for downsampling)  
     - Flatten (to convert 2D → 1D features)  
     - Dense layers (for classification)  
   - Activations:  
     - *ReLU* for hidden layers  
     - *Sigmoid* for the final output layer  

3. *Compilation*  
   - Optimizer: *Adam*  
   - Loss: *Binary Crossentropy* (since binary classification)  

4. *Training*  
   - Trained CNN with fit on images generated from ImageDataGenerator.  
   - Used *4000 images* for training and *1000 images* for testing.  

5. *Prediction on New Images*  
   - Imported image module from Keras preprocessing.  
   - Loaded a new image, converted it to array using img_to_array.  
   - Applied np.expand_dims() to match model input shape.  
   - Final prediction with the trained CNN model.  

## 📂 Code Included
- Image loading with *ImageDataGenerator*  
- CNN model with *Conv2D, MaxPooling2D, Flatten, Dense* layers  
- Compilation with *Adam optimizer* & *Binary Crossentropy loss*  
- Training on Cats vs Dogs dataset  
- Prediction on a new image using image and preprocessing utilities
