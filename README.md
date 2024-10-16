# Brain Tumor Semantic Segmentation

This project implements semantic segmentation of brain tumor cells using a pretrained UNet model in PyTorch. The goal is to accurately segment tumor regions from MRI scans of the brain. The dataset is sourced from Kaggle, and the code is modularized into various scripts for ease of use.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project performs brain tumor segmentation using semantic segmentation techniques with the UNet architecture. The segmentation model was trained on the LGG MRI dataset, enabling the system to identify and segment tumor regions in MRI images of the brain. PyTorch is used as the primary deep learning framework for implementing the model.

## Features
- **Pretrained UNet Model:** Utilizes a pretrained UNet model for performing semantic segmentation.
- **PyTorch Implementation:** Full implementation in PyTorch with easy-to-use scripts for loading data, preprocessing, and training.
- **Modular Code Structure:** The project is divided into multiple scripts for better maintainability and flexibility.
  
## Dataset
The dataset used for this project is the **LGG Segmentation Dataset**, available on Kaggle. It contains MRI scans of the brain along with their corresponding segmentation masks for tumor regions.

Download the dataset from [Kaggle: LGG Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Scripts Overview

- **`data_loaders.py`:** Contains code for loading the dataset and splitting it into training and validation sets.
- **`data_preprocessing.py`:** Handles all data preprocessing tasks like image resizing, normalization, and augmentation.
- **`libraries.py`:** Imports all necessary libraries required throughout the project, making it easier to manage dependencies.
- **`loss_fun.py`:** Defines the loss functions used for model training, such as Dice Loss or Cross-Entropy Loss.
- **`model.py`:** Contains the PyTorch implementation of the UNet model used for segmentation.
- **`engine.py`:** Main script to train and evaluate the model. It loads the data, preprocesses it, and runs the training pipeline.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/brain_tumor_semantic_segmentation.git
    cd brain_tumor_semantic_segmentation
    ```
2. To run the segmentation model, execute the following command:
    ```bash
    python engine.py
    ```

This script will train the model on the dataset, and after training, it will output the performance metrics for the segmentation task.

## Model Architecture

The UNet architecture is a popular choice for image segmentation tasks. It consists of:
- **Encoder:** A series of convolutional layers that capture high-level features from the input image.
- **Decoder:** Symmetric to the encoder, it uses transposed convolutions to upsample the features back to the original resolution while incorporating context from the encoder through skip connections.
- **Skip Connections:** Allows information to pass directly between corresponding layers in the encoder and decoder, improving segmentation accuracy.

## Contributing
Contributions are welcome! If you have suggestions or find any bugs, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
