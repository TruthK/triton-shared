#!/bin/bash

# Ensure the script stops on error
set -e

# Define the environment name
ENV_NAME="triton_shared_mlir_nv"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install conda first."
    exit 1
fi

# Check if the environment already exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Environment $ENV_NAME already exists. Do you want to recreate it? (y/n)"
    read -r answer
    if [[ "$answer" != "y" ]]; then
        echo "Script terminated."
        exit 0
    else
        echo "Removing existing environment $ENV_NAME..."
        conda remove --name "$ENV_NAME" --all -y || { echo "Failed to remove existing environment."; exit 1; }
    fi
fi

# Create a new Conda environment and install all dependencies in two steps
echo "Creating a new Conda environment: $ENV_NAME and installing dependencies..."
conda create -n "$ENV_NAME" python=3.10  gcc libgcc libstdcxx-ng cmake pybind11 ninja pytest pandas matplotlib setuptools requests numpy wheel sympy -y || { echo "Failed to create environment and install dependencies."; exit 1; }

# Activate the environment
echo "Activating environment: $ENV_NAME"
source activate "$ENV_NAME" || { echo "Failed to activate environment."; exit 1; }

# Completion message
echo "Environment $ENV_NAME has been created and all dependencies have been installed successfully!"