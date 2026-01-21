"""
DAgger (Dataset Aggregation) algorithm for imitation learning.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import copy


class DAggerAlgorithm:
    """
    Implements the DAgger algorithm for imitation learning in JSP.
    
    DAgger iteratively:
    1. Trains a policy on the current dataset
    2. Collects new data by executing the learned policy
    3. Labels new states with expert actions
    4. Aggregates new data into the dataset
    """
    
    def __init__(self, expert, feature_extractor, preference_builder, 
                 model, n_iterations: int = 10, beta_decay: float = 0.8):
        """
        Initialize DAgger algorithm.
        
        Args:
            expert: Expert policy (e.g., GurobiExpert)
            feature_extractor: FeatureExtractor instance
            preference_builder: PreferenceBuilder instance
            model: Learning model (e.g., OrdinalRegressionModel)
            n_iterations: Number of DAgger iterations
            beta_decay: Decay rate for expert mixing probability
        """
        self.expert = expert
        self.feature_extractor = feature_extractor
        self.preference_builder = preference_builder
        self.model = model
        self.n_iterations = n_iterations
        self.beta_decay = beta_decay
        self.beta = 1.0  # Initial probability of using expert
        
        self.training_history = []
    
    def run(self, env, n_episodes_per_iter: int = 10) -> Dict:
        """
        Run the DAgger algorithm.
        
        Args:
            env: Environment for rollouts (should have reset() and step() methods)
            n_episodes_per_iter: Number of episodes to collect per iteration
            
        Returns:
            Dictionary with training history and final model
        """
        for iteration in range(self.n_iterations):
            print(f"DAgger iteration {iteration + 1}/{self.n_iterations}")
            
            # Collect data using current policy mixed with expert
            states, actions = self._collect_data(env, n_episodes_per_iter, self.beta)
            
            # Get expert labels for collected states
            expert_actions = self._get_expert_labels(states)
            
            # Build preference pairs from trajectories
            pairs = self.preference_builder.build_pairs_from_trajectory(
                states, expert_actions
            )
            self.preference_builder.add_pairs(pairs)
            
            # Train model on aggregated dataset
            features1, features2, labels = self.preference_builder.get_training_data()
            
            if len(labels) > 0:
                self.model.fit(features1, features2, labels)
                
                # Evaluate current model
                train_accuracy = self.model.score(features1, features2, labels)
                
                self.training_history.append({
                    'iteration': iteration,
                    'n_pairs': len(labels),
                    'train_accuracy': train_accuracy,
                    'beta': self.beta
                })
                
                print(f"  Collected {len(pairs)} new pairs, "
                      f"total {len(labels)} pairs, "
                      f"accuracy: {train_accuracy:.3f}")
            
            # Decay beta (reduce expert influence)
            self.beta *= self.beta_decay
        
        return {
            'history': self.training_history,
            'model': self.model,
            'final_pairs': len(self.preference_builder)
        }
    
    def _collect_data(self, env, n_episodes: int, beta: float) -> tuple:
        """
        Collect state-action pairs using mixed policy.
        
        Args:
            env: Environment for rollouts
            n_episodes: Number of episodes to collect
            beta: Probability of using expert policy
            
        Returns:
            Tuple of (states, actions) lists
        """
        all_states = []
        all_actions = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            
            while not done:
                # Mix expert and learned policy
                if np.random.random() < beta:
                    # Use expert
                    action = self.expert.get_expert_action(state)
                else:
                    # Use learned policy
                    action = self._get_learned_action(state)
                
                all_states.append(copy.deepcopy(state))
                all_actions.append(action)
                
                state, reward, done, info = env.step(action)
        
        return all_states, all_actions
    
    def _get_learned_action(self, state: Dict) -> int:
        """
        Get action from learned policy.
        
        Args:
            state: Current state
            
        Returns:
            Action (job ID)
        """
        available_jobs = state.get('available_jobs', [])
        
        if len(available_jobs) == 1:
            return available_jobs[0]
        
        # Score each available action
        best_job = available_jobs[0]
        best_score = float('-inf')
        
        for job in available_jobs:
            state_copy = state.copy()
            state_copy['candidate_job'] = job
            features = self.feature_extractor.extract_features(state_copy)
            
            # Use model to predict preference score
            score = self.model.predict_score(features.reshape(1, -1))[0]
            
            if score > best_score:
                best_score = score
                best_job = job
        
        return best_job
    
    def _get_expert_labels(self, states: List[Dict]) -> List[int]:
        """
        Get expert action labels for states.
        
        Args:
            states: List of states
            
        Returns:
            List of expert actions
        """
        expert_actions = []
        
        for state in states:
            action = self.expert.get_expert_action(state)
            expert_actions.append(action)
        
        return expert_actions
    
    def save_history(self, filepath: str):
        """
        Save training history to file.
        
        Args:
            filepath: Path to save history
        """
        import csv
        
        if not self.training_history:
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
            writer.writeheader()
            writer.writerows(self.training_history)
