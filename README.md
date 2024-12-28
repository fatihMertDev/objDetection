# objDetection

This repository implements a custom object detection model using Faster R-CNN with a ResNet50 backbone, pre-trained on ImageNet. While a similar version of this model is available in PyTorch's official tutorials, this implementation explores a unique approach by modifying the backbone architecture, optimizing the anchor generation, and adjusting key configurations. The model was trained and tested on the Penn-Fudan Pedestrian dataset, which contains labeled images of pedestrians.

Current Progress
The current implementation outputs results in tensor format. This is suitable for model evaluation and debugging during the early stages of development.

Future Plans

Visual Output: I plan to visualize the detection results by overlaying bounding boxes on the test images for better interpretability.
TensorBoard Integration: To enhance tracking of training metrics and model performance, I aim to integrate TensorBoard for visualizing losses, accuracy, and other key metrics during training.

