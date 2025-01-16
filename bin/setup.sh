#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Installing dependencies..."
pip install -r ../requirements.txt

echo "Cleaning old build artifacts..."
rm -rf ../src/build/ ../src/dist/ ../src/*.egg-info

echo "Building the source distribution and wheel..."
pip install -e ../src/

echo "Make directories"
mkdir ../results
mkdir ../results/base_pred
mkdir ../results/csvs
mkdir ../models

echo "Done!"