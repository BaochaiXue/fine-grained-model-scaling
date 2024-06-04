# Model Optimization Toolkit

## Overview
This toolkit comprises several scripts designed to facilitate the selection, testing, and generation of different variants of machine learning models, specifically tailored for scenarios where storage and computational resources are bounded.

## Scripts

### 1. Storage-bound Model Selection
- **File:** `storage_bound_model_selection.py`
- **Purpose:** Selects the optimal models that fit within specified storage constraints while aiming to maintain performance.
- **Usage:** Run this script with desired storage limits as input parameters.

### 2. Model Testing
- **File:** `python_model_test.py`
- **Purpose:** Tests the performance of machine learning models against predefined metrics to ensure they meet the necessary standards.
- **Usage:** This script is usually invoked by other scripts to automate the testing process, but can also be run standalone with specific model paths and test parameters.

### 3. Model Variant Generation
- **File:** `model_variant_generate.py`
- **Purpose:** Generates various model variants through methods like pruning, useful in studying the effects of different architectural changes on model performance.
- **Usage:** Provide the base model and desired modifications (e.g., pruning levels) as arguments.

### 4. Main Execution Script
- **File:** `main.py`
- **Purpose:** Serves as the entry point for executing the full model optimization workflow, integrating model selection, testing, and generation.
- **Usage:** Run this script to execute the entire optimization process in a step-by-step manner.


### 5. Acknowledgements
The development of this toolkit would not have been possible without the invaluable resources and support from the open-source community. We would like to extend our deepest gratitude to the contributors of the Torch-Pruning library. Their dedication and expertise have provided essential tools and documentation that have significantly enhanced the capabilities of our model optimization processes.

Torch-Pruning has enabled us to implement sophisticated pruning techniques that are critical for generating efficient model variants. The comprehensive documentation and robust functionalities of Torch-Pruning have been instrumental in achieving our goals of optimizing machine learning models within constrained environments. We would also like to acknowledge the broader PyTorch community for their continuous efforts in developing and maintaining an excellent deep learning framework. Their collective work has created a solid foundation upon which we could build and extend our toolkit.

To the researchers, developers, and maintainers of these invaluable resources, we express our sincere appreciation. Your contributions have been pivotal in advancing our work and the field of model optimization.