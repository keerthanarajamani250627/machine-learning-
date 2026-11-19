# Traffic Sign Recognition Using Deep Learning

A deep-learning project that classifies German traffic sign images into **43 categories** using a Convolutional Neural Network (CNN) built with Keras and TensorFlow.

## Project Overview

Traffic sign recognition is an important component of intelligent and autonomous driving systems. This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to train a CNN that identifies the class of a traffic sign from an image.

The notebook explores the dataset, preprocesses the images, trains and evaluates a CNN, and then uses data augmentation to improve classification performance.

## Features

- Loads and explores traffic sign images from the GTSRB dataset.
- Preprocesses images using HSV color conversion and histogram equalization.
- Crops and resizes images to **48 × 48** pixels.
- Trains a CNN with six convolutional layers.
- Uses dropout to help reduce overfitting.
- Applies data augmentation, including shifts, zoom, shear, and rotation.
- Evaluates the model on the test dataset and plots accuracy and loss curves.

## Technologies and Libraries

- Python
- Jupyter Notebook
- TensorFlow and Keras
- NumPy
- Matplotlib
- scikit-image
- pandas
- scikit-learn
- h5py

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The notebook expects the dataset to be placed in the following directory structure:

```text
project-folder/
├── Traffic-Sign-Recognition.ipynb
└── data/
    └── traffic_sign_dataset/
        ├── Final_Training/
        │   └── Images/
        │       ├── 00000/
        │       ├── 00001/
        │       └── ...
        ├── Final_Test/
        │   └── Images/
        └── GT-final_test.csv
```

The training images are organized into class folders. The test labels are read from `GT-final_test.csv`.

**Dataset note:** The dataset is not included in this repository/notebook. Obtain it from the official GTSRB source and place the extracted files in the structure shown above.

## How It Works

### 1. Dataset exploration

The notebook loads the training image paths and displays sample traffic sign images.

### 2. Image preprocessing

Each image is:
1. Converted from RGB to HSV.
2. Histogram-equalized on the value (brightness) channel.
3. Center-cropped to a square.
4. Resized to 48 × 48 pixels.

Class labels are converted to one-hot encoded vectors for training.

### 3. CNN model

The sequential CNN contains:
- Convolutional layers with 32, 64, and 128 filters.
- ReLU activation functions.
- Max-pooling layers.
- Dropout layers for regularization.
- A flatten layer.
- A 512-unit dense hidden layer.
- A 43-unit softmax output layer.

The model is compiled with categorical cross-entropy loss and stochastic gradient descent (SGD) with momentum.

### 4. Training

The notebook uses a batch size of **32** and trains for up to **30 epochs**. It also configures a learning-rate scheduler, model checkpointing, and early stopping.

### 5. Data augmentation

A second training stage applies random width and height shifts, zoom, shear, and rotation to training images.

### 6. Evaluation

The notebook evaluates predictions against the test labels and plots training and validation accuracy and loss.

The notebook's conclusion reports:
- **97.3%** test accuracy before data augmentation.
- **98.4%** test accuracy after data augmentation.

These are the results stated in the notebook; actual results may vary depending on the environment, dataset, and training run.

## Getting Started

### Prerequisites

Install Python and Jupyter Notebook, then install the libraries used by the notebook:

```bash
pip install numpy matplotlib scikit-image pandas scikit-learn h5py tensorflow
```

Keras is used by the notebook and may be available through the installed TensorFlow distribution. Depending on the versions installed, you may need to adjust the notebook's Keras imports.

### Run the project

1. Clone or download this project.
2. Download and extract the GTSRB dataset.
3. Place the dataset under `data/traffic_sign_dataset/` using the directory structure above.
4. Install the required Python libraries.
5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Traffic-Sign-Recognition.ipynb`.
7. Run the notebook cells in order.

During training, the notebook saves model checkpoints as `model.h5` and `model_aug.h5` in the working directory.

## Important Compatibility Note

The notebook uses some older Keras APIs and metric names, including `fit_generator`, `predict_classes`, `lr`, and history keys such as `acc` and `val_acc`. These may not work in current TensorFlow/Keras releases. If you encounter compatibility errors, update the notebook to the equivalent APIs supported by your installed version (for example, `model.fit`, `model.predict` followed by `argmax`, and `accuracy`/`val_accuracy` metric names).

## Project Structure

```text
.
├── Traffic-Sign-Recognition.ipynb   # Main project notebook
├── README.md                        # Project documentation
└── data/
    └── traffic_sign_dataset/        # GTSRB dataset (download separately)
```

## References

1. [Keras Tutorial: Traffic Sign Recognition](https://chsasank.github.io/keras-tutorial.html)
2. [Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition](http://image.diku.dk/igel/paper/MvCBMLAfTSR.pdf)
3. [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)

## Conclusion

This project demonstrates how a Convolutional Neural Network can be used to recognize and classify German traffic signs. Image preprocessing prepares the dataset for training, while data augmentation helps improve the model's reported test accuracy.
