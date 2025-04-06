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

## Dataset

This project uses the [Smart Grid Stability Dataset](https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability) from Kaggle. The dataset contains electrical grid stability simulations with the following characteristics:

### Dataset Features:
- `tau1` to `tau4`: Reaction time of participants (seconds)
- `p1` to `p4`: Power consumed by participants (MW)
- `g1` to `g4`: Price elasticity coefficients
- `stab`: Grid stability ('stable' or 'unstable')
- `stabf`: Stability probability (0 to 1)

### Dataset Statistics:
- Total Instances: 60,000
- Features: 12 input variables + 2 output variables
- Size: ~14MB
- Format: CSV file

The dataset was created using mathematical modeling of electrical grid stability, considering various factors such as power consumption, pricing coefficients, and reaction times of different network participants.

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

4. Download the dataset:
   - Visit [Smart Grid Stability Dataset](https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability) on Kaggle
   - Download the `Data_for_UCI_named.csv` file
   - Place it in the project root directory as `data.csv`

## Usage

1. Ensure the dataset is placed in the project directory as `data.csv`
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

## Data Processing

The data pipeline includes:
1. Loading the CSV dataset
2. Preprocessing numerical features (tau1-tau4, p1-p4, g1-g4)
3. Encoding the target variable (stab)
4. Scaling features for optimal model performance
5. Splitting data into training and testing sets

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset provided by Paulo Cortez and Guilherme Pereira via [UCI Machine Learning Repository](https://archive.ics.uci.edu/) 