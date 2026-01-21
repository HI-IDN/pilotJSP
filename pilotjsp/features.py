"""
Dispatching rule feature extraction for job-shop scheduling.

Extracts 13 features commonly used in dispatching rules:
1. Processing time (PT)
2. Remaining processing time (RPT)
3. Number of remaining operations (NRO)
4. Work remaining (WR)
5. Slack time (ST)
6. Critical ratio (CR)
7. Shortest processing time (SPT)
8. Earliest due date (EDD)
9. First come first served (FCFS)
10. Machine workload (MW)
11. Machine waiting time (MWT)
12. Operation due date (ODD)
13. Total work content (TWC)
"""

import numpy as np
from typing import Dict, List, Optional
import csv


class FeatureExtractor:
    """
    Extracts dispatching rule features from JSP state.
    """
    
    FEATURE_NAMES = [
        'processing_time',
        'remaining_processing_time',
        'num_remaining_operations',
        'work_remaining',
        'slack_time',
        'critical_ratio',
        'shortest_processing_time',
        'earliest_due_date',
        'arrival_time',
        'machine_workload',
        'machine_waiting_time',
        'operation_due_date',
        'total_work_content'
    ]
    
    def __init__(self, instance):
        """
        Initialize feature extractor.
        
        Args:
            instance: JSPInstance object
        """
        self.instance = instance
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
    
    def extract_features(self, state: Dict) -> np.ndarray:
        """
        Extract all 13 features for a given state.
        
        Args:
            state: Dictionary containing:
                - 'job_progress': Array of operation indices completed for each job
                - 'current_time': Current simulation time
                - 'machine_available': Array of next available time for each machine
                - 'job_release': Array of release times for each job
                
        Returns:
            Feature vector of shape (13,) containing normalized features
        """
        features = np.zeros(13)
        
        job_progress = state.get('job_progress', np.zeros(self.n_jobs, dtype=int))
        current_time = state.get('current_time', 0)
        machine_available = state.get('machine_available', np.zeros(self.n_machines))
        job_release = state.get('job_release', np.zeros(self.n_jobs))
        candidate_job = state.get('candidate_job', 0)
        
        # Get current operation for candidate job
        op_id = job_progress[candidate_job]
        
        if op_id >= self.n_machines:
            # Job is complete
            return features
        
        machine, proc_time = self.instance.get_operation(candidate_job, op_id)
        
        # Feature 1: Processing time
        features[0] = proc_time
        
        # Feature 2: Remaining processing time
        remaining_pt = sum(self.instance.processing_times[candidate_job, op_id:])
        features[1] = remaining_pt
        
        # Feature 3: Number of remaining operations
        features[2] = self.n_machines - op_id
        
        # Feature 4: Work remaining (same as feature 2 in this context)
        features[3] = remaining_pt
        
        # Feature 5: Slack time (simplified - due date minus current time minus remaining work)
        # Assuming due date is estimated as total work + release time
        total_work = sum(self.instance.processing_times[candidate_job, :])
        estimated_due = job_release[candidate_job] + total_work * 1.5
        features[4] = max(0, estimated_due - current_time - remaining_pt)
        
        # Feature 6: Critical ratio
        time_to_due = estimated_due - current_time
        if time_to_due > 0:
            features[5] = remaining_pt / time_to_due
        else:
            features[5] = 999  # Very high priority
        
        # Feature 7: Shortest processing time (normalized)
        features[6] = proc_time
        
        # Feature 8: Earliest due date
        features[7] = estimated_due
        
        # Feature 9: Arrival/release time (FCFS)
        features[8] = job_release[candidate_job]
        
        # Feature 10: Machine workload
        features[9] = machine_available[machine]
        
        # Feature 11: Machine waiting time
        features[10] = max(0, current_time - machine_available[machine])
        
        # Feature 12: Operation due date
        ops_remaining = self.n_machines - op_id
        if ops_remaining > 0:
            features[11] = estimated_due - (remaining_pt - proc_time) / ops_remaining
        else:
            features[11] = estimated_due
        
        # Feature 13: Total work content
        features[12] = total_work
        
        return features
    
    def extract_batch_features(self, states: List[Dict]) -> np.ndarray:
        """
        Extract features for multiple states.
        
        Args:
            states: List of state dictionaries
            
        Returns:
            Array of shape (n_states, 13) containing features
        """
        n_states = len(states)
        features = np.zeros((n_states, 13))
        
        for i, state in enumerate(states):
            features[i] = self.extract_features(state)
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range.
        
        Args:
            features: Array of shape (n_samples, 13)
            
        Returns:
            Normalized feature array
        """
        normalized = features.copy()
        
        for i in range(features.shape[1]):
            col = features[:, i]
            min_val = col.min()
            max_val = col.max()
            
            if max_val > min_val:
                normalized[:, i] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0
        
        return normalized
    
    def save_features_to_csv(self, features: np.ndarray, filepath: str):
        """
        Save extracted features to CSV file.
        
        Args:
            features: Feature array of shape (n_samples, 13)
            filepath: Path to save CSV file
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.FEATURE_NAMES)
            
            for feature_row in features:
                writer.writerow(feature_row)
    
    @staticmethod
    def load_features_from_csv(filepath: str) -> np.ndarray:
        """
        Load features from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Feature array
        """
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            features = []
            
            for row in reader:
                features.append([float(x) for x in row])
        
        return np.array(features)
