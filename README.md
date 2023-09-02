## Kaggle Digit Recognizer

![Kaggle Logo](https://storage.googleapis.com/kaggle-competitions/kaggle/3004/logos/header.png)

### Overview

This repository contains code and resources for the Kaggle Digit Recognizer competition. The competition's goal is to correctly identify digits from a dataset of handwritten images ranging from 0 to 9. It serves as an excellent introduction to computer vision techniques and classification algorithms.

For more details about the competition, visit the [Kaggle Digit Recognizer Competition Page](https://www.kaggle.com/competitions/digit-recognizer/overview).

### Getting Started

#### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (for managing Python environments)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) (for data download) or manually download from [here](https://www.kaggle.com/competitions/digit-recognizer/data)

#### Required packages
- Tensorflow
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn


This will download the dataset files into the `data/` directory.

### Project Structure

- `data/`: The location to store the downloaded competition data.
- `environment.yml`: Conda environment configuration file.
- `README.md`: This file providing an overview of the project.
- `utils.py`: Place your code, Jupyter notebooks, or Python scripts here.

### `utils.py`

This Python script contains utility functions and classes for various tasks related to the Kaggle Digit Recognizer project.

#### `PlotLearning` Class

`PlotLearning` is a custom Keras callback used to plot the learning curves of the model during training. It visualizes training and validation metrics over epochs. The key methods and attributes include:

- `on_train_begin(self, logs={})`: Initializes an empty dictionary `self.metrics` to store metrics.
- `on_epoch_end(self, epoch, logs={})`: Appends metrics from each epoch to the `self.metrics` dictionary and plots the learning curves.

#### `plot_confusion_matrix` Function

This function is used to print and plot a confusion matrix for model evaluation. It can optionally normalize the matrix. The key parameters include:

- `cm`: The confusion matrix to be plotted.
- `classes`: The classes (labels) for the confusion matrix.
- `normalize`: A boolean indicating whether to normalize the matrix.
- `title`: The title for the plot.
- `cmap`: The color map for the plot.

#### `plot_count` Function

This function generates a count plot of labels in the training dataset. It annotates each bar with its count. The key parameter is `Y_train`, which represents the training labels.

#### `plot_image_example` Function

This function is used to plot a single image from the dataset. The key parameter is `arr`, which represents the image array to be displayed.

Additionally, the script imports several libraries and modules, including `numpy`, `tensorflow.keras`, `matplotlib.pyplot`, `IPython.display`, `itertools`, and `seaborn`, which are used within these functions for various visualization and data manipulation tasks.

### Usage

`Train_CNN.ipynb` orchestrates a series of functions and steps, including data acquisition, preprocessing, model architecture definition, training, and evaluation, resulting in a streamlined workflow that culminates in a trained and validated CNN model ready for use.






### Acknowledgements

- Kaggle: [Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer/overview)
- [Yassine Ghouzam's Inspiration Kernel](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)

### Contributing

Feel free to contribute to this project. Create pull requests or open issues if you have any improvements or suggestions.

### License

This project is licensed under the [MIT License](LICENSE).

### Contact

For any questions or feedback, please use the [Discussion Forum](https://www.kaggle.com/competitions/digit-recognizer/discussion) on Kaggle.

### Citation

AstroDave, Will Cukierski. (2012). Digit Recognizer. Kaggle. [https://kaggle.com/competitions/digit-recognizer](https://kaggle.com/competitions/digit-recognizer)
