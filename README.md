# Smart Grid Stability Predictor

A machine learning system for predicting power grid stability using advanced ML techniques. This project implements a Random Forest Classifier to analyze and predict the stability of smart power grids based on various input parameters.

## Project Overview

The Smart Grid Stability Predictor is designed to:
- Process and analyze smart grid data
- Train a machine learning model for stability prediction
- Provide real-time predictions for grid stability
- Visualize results and feature importance

## Features

- Advanced data preprocessing and feature scaling
- Random Forest Classification with hyperparameter tuning
- Cross-validation for robust model evaluation
- Feature importance analysis
- Real-time prediction capabilities
- Comprehensive visualization tools

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[username]/capstone.git
   cd capstone
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data in CSV format with the required features
2. Run the predictor:
   ```python
   python smart_grid_stability_predictor.py
   ```

## Model Details

The system uses a Random Forest Classifier with:
- Feature scaling using StandardScaler
- Cross-validation for model validation
- GridSearchCV for hyperparameter optimization
- Comprehensive evaluation metrics

## Data

The model expects a CSV file with the following features:
- Grid stability parameters
- Power consumption metrics
- System state variables

Note: Sample data is not included in the repository due to size constraints.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 