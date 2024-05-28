# Import necessary packages
import torch  # Tensor library
from torch.utils.data import DataLoader  # Data loader
from torchvision import datasets, transforms  # Datasets and data transformations
from ultralytics import YOLO  # YOLO model from ultralytics
import nni  # NNI for hyperparameter tuning
from typing import List, Tuple  # Type hints


# Prepare the CIFAR-10 dataset
def prepare_data(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the CIFAR-10 dataset and return train and test data loaders.

    :param batch_size: Batch size for data loaders.
    :return: Tuple containing train and test data loaders.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the images
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Fine-grained model scaling: Generate model variants
def generate_model_variant(base_model_path: str, scaling_factor: float) -> YOLO:
    """
    Generate a scaled model variant by adjusting the width (number of filters) of convolutional layers.

    :param base_model_path: Path to the base YOLOv8 model file.
    :param scaling_factor: Scaling factor.
    :return: Scaled YOLOv8 model.
    """
    model = YOLO(base_model_path).model  # Load the base model
    for layer in model.model:
        if hasattr(layer, "conv"):
            layer.conv.out_channels = int(
                layer.conv.out_channels * scaling_factor
            )  # Scale the output channels
            layer.conv = torch.nn.Conv2d(
                layer.conv.in_channels,
                layer.conv.out_channels,
                layer.conv.kernel_size,
                layer.conv.stride,
                layer.conv.padding,
            )  # Update the convolutional layer
    return YOLO(model)  # Return the scaled model


# Train the model
def train_model(
    model: YOLO,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    """
    Train the given model on the training dataset.

    :param model: YOLOv8 model to train.
    :param train_loader: Data loader for training data.
    :param num_epochs: Number of epochs for training.
    :param learning_rate: Learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = model.loss(
                outputs, labels
            )  # Compute the loss using the model's built-in loss function
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters
            running_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )
        nni.report_intermediate_result(
            running_loss / len(train_loader)
        )  # Report intermediate result to NNI

    nni.report_final_result(
        running_loss / len(train_loader)
    )  # Report final result to NNI


# Main function to execute the training of model variants
def main() -> None:
    """
    Main function to prepare data, generate model variants, and train them.
    """
    params = nni.get_next_parameter()  # Get the next set of parameters from NNI
    scaling_factor = params.get("scaling_factor", 1.0)  # Default scaling factor is 1.0

    train_loader, test_loader = prepare_data(batch_size=64)  # Prepare data loaders
    base_model_path = "yolov8n.pt"  # Path to the pre-trained YOLOv8 model
    model_variant = generate_model_variant(
        base_model_path, scaling_factor
    )  # Generate model variant

    train_model(
        model_variant, train_loader, num_epochs=10, learning_rate=0.001
    )  # Train the model variant


if __name__ == "__main__":
    main()
