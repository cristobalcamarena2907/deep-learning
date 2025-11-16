#!/bin/bash
# Setup script for UCF101 Action Recognition Project

echo "Setting up UCF101 Action Recognition Project..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p results
mkdir -p logs
mkdir -p notebooks

# Check if data file exists
if [ -f "data/UCF101 Module 2 Deep Learning.pkl" ]; then
    echo "✓ Data file found"
else
    echo "⚠ Warning: Data file not found at data/UCF101 Module 2 Deep Learning.pkl"
    echo "  Please make sure to place the pickle file in the data/ directory"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train the baseline model:"
echo "  python src/train.py --model baseline --split train1 --epochs 50"
echo ""
echo "To train the ST-GCN model:"
echo "  python src/train.py --model st_gcn --split train1 --epochs 100"

