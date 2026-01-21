"""
Preference pair builder for imitation learning.
"""

import numpy as np
from typing import List, Tuple, Dict
import csv


class PreferenceBuilder:
    """
    Builds preference pairs from expert demonstrations for learning.
    """
    
    def __init__(self, feature_extractor, expert):
        """
        Initialize preference builder.
        
        Args:
            feature_extractor: FeatureExtractor instance
            expert: Expert policy (e.g., GurobiExpert)
        """
        self.feature_extractor = feature_extractor
        self.expert = expert
        self.preference_pairs = []
    
    def build_pairs_from_trajectory(self, states: List[Dict], 
                                   actions: List[int]) -> List[Tuple]:
        """
        Build preference pairs from a trajectory of states and expert actions.
        
        Args:
            states: List of state dictionaries
            actions: List of expert action choices
            
        Returns:
            List of preference pairs (features_preferred, features_other, label)
        """
        pairs = []
        
        for state, expert_action in zip(states, actions):
            available_jobs = state.get('available_jobs', [])
            
            if len(available_jobs) < 2:
                continue
            
            # Extract features for expert's choice
            state_expert = state.copy()
            state_expert['candidate_job'] = expert_action
            features_expert = self.feature_extractor.extract_features(state_expert)
            
            # Create preference pairs with other available actions
            for job in available_jobs:
                if job != expert_action:
                    state_other = state.copy()
                    state_other['candidate_job'] = job
                    features_other = self.feature_extractor.extract_features(state_other)
                    
                    # Label: 1 if expert's choice is preferred, -1 otherwise
                    pairs.append((features_expert, features_other, 1))
                    pairs.append((features_other, features_expert, -1))
        
        return pairs
    
    def build_pairs_from_rollouts(self, initial_states: List[Dict], 
                                  n_rollouts: int = 10) -> List[Tuple]:
        """
        Build preference pairs by comparing rollout qualities.
        
        Args:
            initial_states: List of initial states to roll out from
            n_rollouts: Number of rollouts per state
            
        Returns:
            List of preference pairs
        """
        all_pairs = []
        
        for state in initial_states:
            # Get expert's recommended action
            expert_action = self.expert.get_expert_action(state)
            available_jobs = state.get('available_jobs', [])
            
            # Extract features for all available actions
            action_features = {}
            for job in available_jobs:
                state_copy = state.copy()
                state_copy['candidate_job'] = job
                action_features[job] = self.feature_extractor.extract_features(state_copy)
            
            # Create preference pairs based on expert choice
            for job in available_jobs:
                if job != expert_action:
                    pairs = [
                        (action_features[expert_action], action_features[job], 1),
                        (action_features[job], action_features[expert_action], -1)
                    ]
                    all_pairs.extend(pairs)
        
        return all_pairs
    
    def add_pairs(self, pairs: List[Tuple]):
        """
        Add preference pairs to the collection.
        
        Args:
            pairs: List of (features1, features2, label) tuples
        """
        self.preference_pairs.extend(pairs)
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data in format suitable for learning.
        
        Returns:
            Tuple of (features1, features2, labels)
        """
        if not self.preference_pairs:
            return np.array([]), np.array([]), np.array([])
        
        n_pairs = len(self.preference_pairs)
        n_features = len(self.preference_pairs[0][0])
        
        features1 = np.zeros((n_pairs, n_features))
        features2 = np.zeros((n_pairs, n_features))
        labels = np.zeros(n_pairs)
        
        for i, (f1, f2, label) in enumerate(self.preference_pairs):
            features1[i] = f1
            features2[i] = f2
            labels[i] = label
        
        return features1, features2, labels
    
    def save_to_csv(self, filepath: str):
        """
        Save preference pairs to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.preference_pairs:
            return
        
        n_features = len(self.preference_pairs[0][0])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = []
            for i in range(n_features):
                header.append(f'feature1_{i}')
            for i in range(n_features):
                header.append(f'feature2_{i}')
            header.append('label')
            writer.writerow(header)
            
            # Data
            for f1, f2, label in self.preference_pairs:
                row = list(f1) + list(f2) + [label]
                writer.writerow(row)
    
    def load_from_csv(self, filepath: str):
        """
        Load preference pairs from CSV file.
        
        Args:
            filepath: Path to CSV file
        """
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            n_features = (len(header) - 1) // 2
            
            self.preference_pairs = []
            for row in reader:
                f1 = np.array([float(x) for x in row[:n_features]])
                f2 = np.array([float(x) for x in row[n_features:2*n_features]])
                label = float(row[-1])
                self.preference_pairs.append((f1, f2, label))
    
    def clear(self):
        """Clear all stored preference pairs."""
        self.preference_pairs = []
    
    def __len__(self):
        """Return number of preference pairs."""
        return len(self.preference_pairs)
