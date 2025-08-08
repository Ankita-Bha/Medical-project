# Medical Imaging Organ Classification Project

A deep learning project for classifying medical images (DICOM files) based on body parts examined, specifically focused on chest CT scans. This project uses Convolutional Neural Networks (CNN) to automatically classify medical images.

## 📋 Project Overview

This project implements a machine learning system for medical image classification using DICOM (Digital Imaging and Communications in Medicine) files. The system is designed to classify CT scan images based on the body part examined, with a primary focus on chest imaging.

## 🎯 Features

- **DICOM Image Processing**: Loads and processes medical DICOM files
- **CNN-based Classification**: Uses Convolutional Neural Networks for image classification
- **Multi-class Support**: Can handle multiple body part classifications
- **Data Preprocessing**: Includes image resizing, normalization, and augmentation
- **Model Training & Evaluation**: Complete training pipeline with performance metrics
- **Overfitting Analysis**: Includes tools for detecting and preventing overfitting

## 📁 Project Structure

```
Medical-project/
├── dicom_cnn_starter.ipynb          # Main CNN implementation notebook
├── BodyPartExamined.ipynb           # Body part classification notebook
├── overfitting_test.ipynb           # Overfitting analysis notebook
├── scan_inventory.csv               # Dataset metadata and labels
├── label_classes.json               # Class labels configuration
├── label_classes.npy                # NumPy array of class labels
├── dicom_organ_classifier.h5        # Trained model (HDF5 format)
├── dicom_organ_classifier.keras     # Trained model (Keras format)
├── my_model.h5                      # Additional trained model
├── history.npy                      # Training history data
├── X_test.npy                       # Test dataset features
├── y_test.npy                       # Test dataset labels
├── dicom_data/                      # DICOM files directory
├── dicom_data_bulk/                 # Bulk DICOM data directory
├── exported_chest_jpg/              # Exported chest images (JPG)
└── jpg_output/                      # Output JPG images
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- PyDICOM
- OpenCV
- NumPy
- Pandas
- Scikit-learn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Medical-project
```

2. Install required packages:
```bash
pip install tensorflow pydicom opencv-python numpy pandas scikit-learn
```

3. Ensure your DICOM data is organized in the `dicom_data_bulk/` directory with the following structure:
```
dicom_data_bulk/
├── SeriesInstanceUID_1/
│   ├── image1.dcm
│   ├── image2.dcm
│   └── ...
├── SeriesInstanceUID_2/
│   ├── image1.dcm
│   └── ...
└── ...
```

## 📊 Dataset

The project uses the CPTAC-LSCC (Clinical Proteomic Tumor Analysis Consortium - Lung Squamous Cell Carcinoma) dataset, which contains:

- **Modality**: CT (Computed Tomography)
- **Body Part**: Primarily CHEST
- **Image Format**: DICOM
- **Metadata**: Available in `scan_inventory.csv`

### Dataset Statistics

- Total series: 215 (from scan_inventory.csv)
- Primary body part: CHEST
- Image size: Variable (resized to 128x128 for training)
- Classes: Currently configured for chest classification

## 🧠 Model Architecture

The CNN model used in this project has the following architecture:

```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### Model Parameters

- **Input Size**: 128x128 pixels (grayscale)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## 📈 Usage

### 1. Data Preparation

Run the data preparation cells in `BodyPartExamined.ipynb`:

```python
# Load and preprocess DICOM data
df = pd.read_csv("scan_inventory.csv")
# ... data preprocessing steps
```

### 2. Model Training

Execute the training cells in `dicom_cnn_starter.ipynb`:

```python
# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

### 3. Model Evaluation

Evaluate the trained model:

```python
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### 4. Overfitting Analysis

Use `overfitting_test.ipynb` to analyze model performance and detect overfitting.

## 💾 Model Files

The project includes several trained models:

- `dicom_organ_classifier.h5` (85MB): Main trained model
- `dicom_organ_classifier.keras` (370KB): Keras format model
- `my_model.h5` (75MB): Alternative trained model

## 🔧 Configuration

### Image Processing Settings

- **Image Size**: 128x128 pixels
- **Normalization**: Pixel values scaled to [0,1]
- **Data Augmentation**: Available for training

### Training Parameters

- **Test Split**: 20% of data
- **Random State**: 42 (for reproducibility)
- **Stratification**: Enabled for balanced splits

## 📊 Results

The model achieves classification accuracy on chest CT scans. Performance metrics are saved in:

- `history.npy`: Training history
- `X_test.npy` & `y_test.npy`: Test dataset for evaluation

## 🛠️ Customization

### Adding New Classes

1. Update `label_classes.json` with new class names
2. Modify the data loading logic in the notebooks
3. Adjust the model's final layer for the new number of classes

### Changing Image Size

Update the `IMG_SIZE` parameter in the notebooks:

```python
IMG_SIZE = (256, 256)  # Change to desired size
```

### Model Architecture Modifications

Modify the CNN architecture in the notebooks to experiment with different configurations.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project uses data from the CPTAC-LSCC dataset, which is licensed under the Creative Commons Attribution 3.0 Unported License.

## ⚠️ Disclaimer

This project is for educational and research purposes. The models and results should not be used for clinical decision-making without proper validation and regulatory approval.

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This project requires appropriate medical imaging data and should be used in compliance with relevant healthcare data regulations and privacy laws. 