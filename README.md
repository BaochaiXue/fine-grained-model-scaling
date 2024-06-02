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


