"""
Ordinal regression model for learning from preference pairs.
"""

import numpy as np
from typing import Optional, Tuple
import pickle


class OrdinalRegressionModel:
    """
    Ordinal regression model for learning from preference pairs.
    
    Uses a simple linear model with ranking loss for preference learning.
    """
    
    def __init__(self, n_features: int = 13, learning_rate: float = 0.01,
                 n_epochs: int = 100, regularization: float = 0.01):
        """
        Initialize ordinal regression model.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for gradient descent
            n_epochs: Number of training epochs
            regularization: L2 regularization coefficient
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization
        
        # Initialize weights
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        self.training_losses = []
    
    def _score_function(self, features: np.ndarray) -> np.ndarray:
        """
        Compute score for features.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Scores of shape (n_samples,)
        """
        return np.dot(features, self.weights) + self.bias
    
    def fit(self, features1: np.ndarray, features2: np.ndarray, 
            labels: np.ndarray, verbose: bool = False):
        """
        Fit the model using preference pairs.
        
        Args:
            features1: Features for first option (n_samples, n_features)
            features2: Features for second option (n_samples, n_features)
            labels: Preference labels (1 if option1 > option2, -1 otherwise)
            verbose: Whether to print training progress
        """
        n_samples = len(labels)
        
        for epoch in range(self.n_epochs):
            # Compute scores
            scores1 = self._score_function(features1)
            scores2 = self._score_function(features2)
            
            # Compute preference differences
            score_diff = scores1 - scores2
            
            # Hinge loss for preferences: max(0, -label * score_diff + margin)
            margin = 1.0
            losses = np.maximum(0, -labels * score_diff + margin)
            
            # Average loss with regularization
            loss = np.mean(losses) + self.regularization * np.sum(self.weights ** 2)
            self.training_losses.append(loss)
            
            # Compute gradients
            # Gradient is non-zero only where loss is active
            active = losses > 0
            n_active = np.sum(active)
            
            if n_active > 0:
                # Gradient of hinge loss
                grad_weights = np.zeros(self.n_features)
                grad_bias = 0.0
                
                for i in range(n_samples):
                    if active[i]:
                        grad_weights += -labels[i] * (features1[i] - features2[i])
                        grad_bias += -labels[i]
                
                grad_weights = grad_weights / n_samples + 2 * self.regularization * self.weights
                grad_bias = grad_bias / n_samples
                
                # Update parameters
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")
    
    def predict_score(self, features: np.ndarray) -> np.ndarray:
        """
        Predict scores for features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted scores (n_samples,)
        """
        return self._score_function(features)
    
    def predict_preference(self, features1: np.ndarray, 
                          features2: np.ndarray) -> np.ndarray:
        """
        Predict preference between two options.
        
        Args:
            features1: Features for first option
            features2: Features for second option
            
        Returns:
            Preference predictions (1 if option1 > option2, -1 otherwise)
        """
        scores1 = self._score_function(features1)
        scores2 = self._score_function(features2)
        
        return np.sign(scores1 - scores2)
    
    def score(self, features1: np.ndarray, features2: np.ndarray, 
              labels: np.ndarray) -> float:
        """
        Compute accuracy on preference pairs.
        
        Args:
            features1: Features for first option
            features2: Features for second option
            labels: True preference labels
            
        Returns:
            Accuracy (fraction of correct predictions)
        """
        predictions = self.predict_preference(features1, features2)
        accuracy = np.mean(predictions == labels)
        return accuracy
    
    def rank_options(self, feature_list: list) -> np.ndarray:
        """
        Rank multiple options by their scores.
        
        Args:
            feature_list: List of feature arrays
            
        Returns:
            Indices sorted by predicted score (descending)
        """
        features = np.array(feature_list)
        scores = self.predict_score(features)
        ranked_indices = np.argsort(-scores)  # Descending order
        return ranked_indices
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'n_features': self.n_features,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'regularization': self.regularization,
            'training_losses': self.training_losses
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.n_features = model_data['n_features']
        self.learning_rate = model_data.get('learning_rate', 0.01)
        self.n_epochs = model_data.get('n_epochs', 100)
        self.regularization = model_data.get('regularization', 0.01)
        self.training_losses = model_data.get('training_losses', [])
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (absolute weight values).
        
        Returns:
            Array of feature importances
        """
        return np.abs(self.weights)
