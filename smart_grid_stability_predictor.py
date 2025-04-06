import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartGridStabilityPredictor:
    """
    A class for predicting smart grid stability using machine learning.
    
    This implementation includes:
    - Advanced data preprocessing
    - Feature scaling
    - Cross-validation
    - Hyperparameter tuning
    - Model evaluation
    - Feature importance analysis
    - Real-time prediction capabilities
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the smart grid data.
        
        Args:
            data_path (str): Path to the CSV data file
            
        Returns:
            tuple: Preprocessed features and target variables
        """
        try:
            logger.info("Loading data from %s", data_path)
            df = pd.read_csv(data_path)
            
            # Store feature names
            self.feature_names = [col for col in df.columns if col not in ['stab', 'stabf']]
            
            # Prepare features and target
            X = df.drop(columns=['stab', 'stabf'])
            y = self.label_encoder.fit_transform(df['stabf'])
            
            logger.info("Data loaded successfully. Shape: %s", X.shape)
            return X, y
            
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise
            
    def create_and_tune_model(self, X, y):
        """
        Create and tune the Random Forest model using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('classifier', RandomForestClassifier(random_state=self.random_state))
            ])
            
            # Define a smaller parameter grid for initial testing
            param_grid = {
                'classifier__n_estimators': [100],  # Reduced from [100, 200, 300]
                'classifier__max_depth': [10, 15],  # Reduced from [10, 15, 20]
                'classifier__min_samples_split': [2, 5],  # Reduced from [2, 5, 10]
                'classifier__min_samples_leaf': [1]  # Reduced from [1, 2, 4]
            }
            
            # Perform grid search with verbose output
            logger.info("Starting hyperparameter tuning with reduced parameter grid...")
            print("This may take a few minutes. Progress will be shown...")
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2  # Added verbose output
            )
            
            print("Fitting model with data shape:", X.shape)
            grid_search.fit(X, y)
            
            logger.info("Best parameters: %s", grid_search.best_params_)
            logger.info("Best cross-validation score: %.3f", grid_search.best_score_)
            
            self.model = grid_search.best_estimator_
            print("Model training completed successfully!")
            
        except Exception as e:
            logger.error("Error in model tuning: %s", str(e))
            raise
            
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model using various metrics and visualizations.
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()
            
            # Plot feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance in Smart Grid Stability Prediction')
            plt.show()
            
        except Exception as e:
            logger.error("Error in model evaluation: %s", str(e))
            raise
            
    def save_model(self, model_path='smart_grid_model.joblib'):
        """
        Save the trained model and preprocessing components.
        
        Args:
            model_path (str): Path to save the model
        """
        try:
            joblib.dump(self.model, model_path)
            logger.info("Model saved successfully to %s", model_path)
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise
            
    def load_model(self, model_path='smart_grid_model.joblib'):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully from %s", model_path)
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
            
    def suggest_optimization_strategy(self, current_loads, node_stability):
        """
        Analyze grid stability and suggest optimization strategies based on node loads.
        
        Args:
            current_loads (dict): Dictionary of node names and their load percentages
            node_stability (dict): Dictionary of node names and their stability predictions
            
        Returns:
            dict: Optimization suggestions and load redistribution recommendations
        """
        try:
            optimization_suggestions = {
                'overloaded_nodes': [],
                'underutilized_nodes': [],
                'redistribution_suggestions': [],
                'risk_level': 'LOW'
            }
            
            # Analyze node loads and stability
            for node, load in current_loads.items():
                is_stable = node_stability.get(node, True)  # Default to True if not in dict
                
                if load > 80:
                    optimization_suggestions['overloaded_nodes'].append({
                        'node': node,
                        'load': load,
                        'status': 'UNSTABLE' if not is_stable else 'STABLE'
                    })
                    if not is_stable:
                        optimization_suggestions['risk_level'] = 'HIGH'
                
                elif load < 50:
                    optimization_suggestions['underutilized_nodes'].append({
                        'node': node,
                        'load': load,
                        'status': 'UNSTABLE' if not is_stable else 'STABLE'
                    })
            
            # Generate redistribution suggestions
            if optimization_suggestions['overloaded_nodes'] and optimization_suggestions['underutilized_nodes']:
                for overloaded in optimization_suggestions['overloaded_nodes']:
                    for underutilized in optimization_suggestions['underutilized_nodes']:
                        if overloaded['status'] == 'UNSTABLE':
                            load_to_redistribute = min(
                                overloaded['load'] - 70,  # Target 70% load
                                50 - underutilized['load']  # Available capacity
                            )
                            if load_to_redistribute > 0:
                                optimization_suggestions['redistribution_suggestions'].append({
                                    'from_node': overloaded['node'],
                                    'to_node': underutilized['node'],
                                    'load_to_move': round(load_to_redistribute, 2),
                                    'priority': 'HIGH' if overloaded['status'] == 'UNSTABLE' else 'MEDIUM'
                                })
            
            return optimization_suggestions
            
        except Exception as e:
            logger.error("Error in optimization strategy: %s", str(e))
            raise

    def visualize_grid_status(self, current_loads, node_stability, save_path=None):
        """
        Visualize the grid nodes with their load status and stability.
        
        Args:
            current_loads (dict): Dictionary of node names and their load percentages
            node_stability (dict): Dictionary of node names and their stability predictions
            save_path (str, optional): Path to save the visualization
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Create a grid layout
            n_nodes = len(current_loads)
            grid_size = int(np.ceil(np.sqrt(n_nodes)))
            
            for idx, (node, load) in enumerate(current_loads.items()):
                is_stable = node_stability.get(node, True)
                
                # Calculate position in grid
                row = idx // grid_size
                col = idx % grid_size
                
                # Create node circle
                circle = plt.Circle((col, row), 0.4, 
                                  color='green' if is_stable else 'red',
                                  alpha=0.6)
                plt.gca().add_patch(circle)
                
                # Add node label and load
                plt.text(col, row, f'{node}\n{load}%', 
                        ha='center', va='center',
                        color='black' if is_stable else 'white')
            
            plt.xlim(-0.5, grid_size)
            plt.ylim(-0.5, grid_size)
            plt.title('Smart Grid Node Status')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.6, label='Stable'),
                Patch(facecolor='red', alpha=0.6, label='Unstable')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Grid visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error("Error in grid visualization: %s", str(e))
            raise

    def predict_stability(self, new_data):
        """
        Make predictions on new data and provide optimization suggestions if needed.
        
        Args:
            new_data: Features for prediction
            
        Returns:
            dict: Prediction results including stability, confidence, and optimization suggestions if unstable
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                raise ValueError("Model not trained or loaded. Please train or load the model first.")
            
            # Make prediction
            prediction = self.model.predict(new_data)
            prediction_proba = self.model.predict_proba(new_data)
            confidence = np.max(prediction_proba, axis=1)[0]
            
            # Create timestamp
            timestamp = datetime.now().isoformat()
            
            # Base result
            result = {
                'stability': 'stable' if prediction[0] == 1 else 'unstable',
                'confidence': float(confidence),
                'timestamp': timestamp
            }
            
            # If unstable, simulate current loads and get optimization suggestions
            if prediction[0] == 0:  # Unstable prediction
                # Simulate current loads based on features
                # Note: This is a simplified example - in real world, you'd get actual load data
                current_loads = {
                    f'Node_{i+1}': min(100, max(0, float(val * 100))) 
                    for i, val in enumerate(new_data.iloc[0])
                }
                
                # Create node stability dict based on loads
                node_stability = {
                    node: load <= 80  # Consider nodes with >80% load as potentially unstable
                    for node, load in current_loads.items()
                }
                
                # Get optimization suggestions
                optimization_suggestions = self.suggest_optimization_strategy(
                    current_loads,
                    node_stability
                )
                
                # Add suggestions to result
                result['optimization_suggestions'] = optimization_suggestions
                
                # Generate and save visualization
                viz_path = f'grid_status_{timestamp.replace(":", "-")}.png'
                self.visualize_grid_status(current_loads, node_stability, save_path=viz_path)
                result['visualization_path'] = viz_path
                
                # Log the unstable state and suggestions
                logger.warning("Unstable grid state detected!")
                logger.info("Optimization suggestions generated: %s", optimization_suggestions)
            
            return result
            
        except Exception as e:
            logger.error("Error in prediction: %s", str(e))
            raise

def main():
    """
    Main function to demonstrate the usage of the SmartGridStabilityPredictor class.
    """
    try:
        # Initialize predictor
        predictor = SmartGridStabilityPredictor()
        
        # Load and preprocess data
        X, y = predictor.load_and_preprocess_data("data.csv")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        
        # Create and tune model
        predictor.create_and_tune_model(X_train, y_train)
        
        # Evaluate model
        predictor.evaluate_model(X_test, y_test)
        
        # Save model
        predictor.save_model()
        
        # Example prediction
        sample_data = X_test.iloc[[0]]
        result = predictor.predict_stability(sample_data)
        print("\nSample Prediction:")
        print(f"Stability: {result['stability']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Timestamp: {result['timestamp']}")
        
    except Exception as e:
        logger.error("Error in main execution: %s", str(e))
        raise

if __name__ == "__main__":
    main() 