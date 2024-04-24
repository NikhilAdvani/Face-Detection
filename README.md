# Deep Learning Real-time Face Detection and Tracking

This project utilizes deep learning techniques implemented in TensorFlow to detect and track human faces in real-time using a webcam. The system is built on the Functional API of TensorFlow and employs a VGG16 convolutional neural network as its backbone for feature extraction.

## Features
- **Face Detection**: Detects faces in real-time using a pre-trained VGG16 model.
- **Real-Time Tracking**: Tracks detected faces in real-time using bounding boxes.

## Applications

- **Facial Sentiment Analysis**: Can be extended to analyze facial expressions for sentiment analysis purposes.
- **Facial Verification**: Can verify individuals' faces against known faces for authentication purposes.
- **Security Systems**: Implementing facial verification for access control and surveillance systems.
- **Interactive Interfaces**: Creating interactive interfaces that respond to user facial expressions.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- TensorFlow
- OpenCV
- Matplotlib
- Albumentations
- LabelMe (for image annotation)

You can install the dependencies using the provided `pip install` command.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-NikhilAdvani/Face-Detection.git
```

2. Navigate to the project directory:

```bash
cd Face-Detection
```

### Usage

1. Run the data collection script to collect images for training

2. Annotate the collected images using LabelMe for labeling.

3. Review the dataset and build an image loading function using TensorFlow.

4. Partition the unaugmented data into train, test, and validation sets.

5. Apply image augmentation on images and labels using Albumentations to increase the diversity of the dataset.

6. Build and run the augmentation pipeline to generate augmented data.

7. Load augmented images into TensorFlow datasets.

8. Prepare labels and load them into TensorFlow datasets.

9. Combine label and image samples to create final datasets.

10. Build the deep learning model using the Functional API, incorporating VGG16 for feature extraction.

11. Define losses and optimizers for the model.

12. Train the neural network model on the prepared datasets.

13. Make predictions on the test set and save the trained model.

14. Implement real-time face detection and tracking using the trained model and OpenCV.


## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Special thanks to Nicholas Renotte for his guidance and tutorials. Visit his [YouTube channel](https://youtube.com/c/nicholasrenotte) for valuable insights into deep learning and computer vision.
