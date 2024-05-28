# YOLOv8 Model Scaling with CIFAR-10 Using NNI

This project demonstrates how to use NNI (Neural Network Intelligence) for fine-grained model scaling of a YOLOv8 model on the CIFAR-10 dataset. The goal is to find the optimal scaling factor for the model to achieve the best performance.

## Project Structure

- `train.py`: The main script that prepares the data, generates model variants, and trains them using NNI.
- `config.yml`: The NNI configuration file for the experiment.
- `search_space.json`: The search space definition for the hyperparameter tuning.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- ultralytics
- NNI

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install the required packages:
    ```bash
    pip install torch torchvision ultralytics nni
    ```

## Prepare the CIFAR-10 Dataset

The CIFAR-10 dataset will be downloaded automatically when you run the training script.

## Running the Experiment

1. **Configure NNI**: Ensure that `config.yml` and `search_space.json` are correctly set up. These files define the experiment's configuration and the search space for hyperparameter tuning.

2. **Start the NNI experiment**:
    ```bash
    nnictl create --config config.yml
    ```

3. **Monitor the experiment**: You can monitor the progress and results of your experiment using the NNI web UI. The web UI URL will be provided in the terminal after starting the experiment.

## Explanation of Files

### `train.py`

This script performs the following tasks:
- Prepares the CIFAR-10 dataset and returns data loaders.
- Generates model variants by scaling the width of the YOLOv8 model's convolutional layers.
- Trains each model variant and reports the results to NNI for hyperparameter tuning.

### `config.yml`

This file contains the NNI experiment configuration:
- `authorName`: Your name.
- `experimentName`: The name of the experiment.
- `trialConcurrency`: Number of trials to run concurrently.
- `maxExperimentDuration`: Maximum duration for the experiment.
- `maxTrialNumber`: Maximum number of trials.
- `trainingService`: The platform for running the trials (local in this case).
- `searchSpaceFile`: Path to the search space file.
- `trialCommand`: The command to run each trial.
- `trialGpuNumber`: Number of GPUs to allocate per trial.
- `tuner`: The tuning algorithm to use (TPE in this case).
- `assessor`: The assessment algorithm to use (Medianstop in this case).

### `search_space.json`

This file defines the search space for the hyperparameter tuning:
- `scaling_factor`: The range and type of the scaling factor to explore.

## Results

The results of the experiment, including the best scaling factor and corresponding model performance, will be available in the NNI web UI. You can also find detailed logs and intermediate results.

## Acknowledgements

- [NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
