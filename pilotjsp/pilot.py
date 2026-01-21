"""
Pilot rollout heuristic for job-shop scheduling.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import copy


class PilotHeuristic:
    """
    Pilot method that uses limited lookahead to make scheduling decisions.
    
    The pilot heuristic:
    1. For each available action, performs a short rollout
    2. Evaluates the quality of resulting partial schedule
    3. Selects the action leading to the best outcome
    """
    
    def __init__(self, instance, feature_extractor, model, 
                 rollout_depth: int = 5, base_policy: Optional[Callable] = None):
        """
        Initialize pilot heuristic.
        
        Args:
            instance: JSPInstance object
            feature_extractor: FeatureExtractor instance
            model: Trained model for action selection
            rollout_depth: Number of steps to look ahead
            base_policy: Base policy for rollouts (default: learned model)
        """
        self.instance = instance
        self.feature_extractor = feature_extractor
        self.model = model
        self.rollout_depth = rollout_depth
        self.base_policy = base_policy or self._default_policy
    
    def select_action(self, state: Dict) -> int:
        """
        Select best action using pilot rollout.
        
        Args:
            state: Current scheduling state
            
        Returns:
            Best action (job ID)
        """
        available_jobs = state.get('available_jobs', [])
        
        if len(available_jobs) == 1:
            return available_jobs[0]
        
        # Evaluate each available action with rollout
        best_job = available_jobs[0]
        best_value = float('-inf')
        
        for job in available_jobs:
            # Simulate taking this action and rolling out
            value = self._evaluate_action(state, job)
            
            if value > best_value:
                best_value = value
                best_job = job
        
        return best_job
    
    def _evaluate_action(self, state: Dict, action: int) -> float:
        """
        Evaluate an action by performing a rollout.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            Estimated value of taking this action
        """
        # Create a copy of state and apply action
        sim_state = copy.deepcopy(state)
        sim_state = self._apply_action(sim_state, action)
        
        # Perform rollout for specified depth
        total_reward = 0.0
        depth = 0
        
        while depth < self.rollout_depth and not self._is_terminal(sim_state):
            # Select next action using base policy
            next_action = self.base_policy(sim_state)
            
            # Apply action and get reward
            sim_state = self._apply_action(sim_state, next_action)
            reward = self._get_reward(sim_state)
            total_reward += reward
            
            depth += 1
        
        # Final evaluation: negative makespan estimate
        makespan_estimate = self._estimate_makespan(sim_state)
        total_reward -= makespan_estimate
        
        return total_reward
    
    def _default_policy(self, state: Dict) -> int:
        """
        Default policy using the learned model.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        available_jobs = state.get('available_jobs', [])
        
        if len(available_jobs) == 1:
            return available_jobs[0]
        
        # Score each job using learned model
        best_job = available_jobs[0]
        best_score = float('-inf')
        
        for job in available_jobs:
            state_copy = state.copy()
            state_copy['candidate_job'] = job
            features = self.feature_extractor.extract_features(state_copy)
            score = self.model.predict_score(features.reshape(1, -1))[0]
            
            if score > best_score:
                best_score = score
                best_job = job
        
        return best_job
    
    def _apply_action(self, state: Dict, action: int) -> Dict:
        """
        Apply an action to the state (simplified simulation).
        
        Args:
            state: Current state
            action: Job to schedule
            
        Returns:
            New state after action
        """
        new_state = copy.deepcopy(state)
        
        job_progress = new_state.get('job_progress', 
                                     np.zeros(self.instance.n_jobs, dtype=int))
        current_time = new_state.get('current_time', 0)
        machine_available = new_state.get('machine_available', 
                                         np.zeros(self.instance.n_machines))
        
        # Get operation details
        op_id = job_progress[action]
        if op_id >= self.instance.n_machines:
            # Job already complete
            return new_state
        
        machine, proc_time = self.instance.get_operation(action, op_id)
        
        # Schedule operation
        start_time = max(current_time, machine_available[machine])
        completion_time = start_time + proc_time
        
        # Update state
        machine_available[machine] = completion_time
        job_progress[action] += 1
        
        new_state['job_progress'] = job_progress
        new_state['current_time'] = completion_time
        new_state['machine_available'] = machine_available
        
        # Update available jobs
        available = []
        for j in range(self.instance.n_jobs):
            if job_progress[j] < self.instance.n_machines:
                available.append(j)
        new_state['available_jobs'] = available
        
        return new_state
    
    def _is_terminal(self, state: Dict) -> bool:
        """Check if state is terminal (all jobs complete)."""
        job_progress = state.get('job_progress', 
                                np.zeros(self.instance.n_jobs, dtype=int))
        return all(progress >= self.instance.n_machines for progress in job_progress)
    
    def _get_reward(self, state: Dict) -> float:
        """
        Compute immediate reward for state.
        
        Simple reward: negative of current time advancement
        """
        return -state.get('current_time', 0) * 0.01
    
    def _estimate_makespan(self, state: Dict) -> float:
        """
        Estimate final makespan from partial schedule.
        
        Args:
            state: Current state
            
        Returns:
            Estimated makespan
        """
        job_progress = state.get('job_progress', 
                                np.zeros(self.instance.n_jobs, dtype=int))
        machine_available = state.get('machine_available', 
                                     np.zeros(self.instance.n_machines))
        
        # Simple estimate: max of machine availability + remaining work
        max_completion = np.max(machine_available)
        
        # Add estimate for remaining work
        for j in range(self.instance.n_jobs):
            remaining_ops = self.instance.n_machines - job_progress[j]
            if remaining_ops > 0:
                remaining_time = sum(self.instance.processing_times[j, job_progress[j]:])
                max_completion = max(max_completion, 
                                   machine_available[self.instance.machine_order[j, job_progress[j]]] 
                                   + remaining_time)
        
        return max_completion
    
    def solve(self, initial_state: Optional[Dict] = None) -> Dict:
        """
        Solve the JSP instance using pilot heuristic.
        
        Args:
            initial_state: Initial state (if None, creates default)
            
        Returns:
            Solution dictionary with makespan and schedule
        """
        if initial_state is None:
            state = {
                'job_progress': np.zeros(self.instance.n_jobs, dtype=int),
                'current_time': 0,
                'machine_available': np.zeros(self.instance.n_machines),
                'available_jobs': list(range(self.instance.n_jobs)),
                'job_release': np.zeros(self.instance.n_jobs)
            }
        else:
            state = copy.deepcopy(initial_state)
        
        schedule = []
        
        while not self._is_terminal(state):
            # Select action using pilot method
            action = self.select_action(state)
            
            # Record decision
            job_progress = state['job_progress']
            op_id = job_progress[action]
            machine, proc_time = self.instance.get_operation(action, op_id)
            
            start_time = max(state['current_time'], 
                           state['machine_available'][machine])
            
            schedule.append({
                'job': action,
                'operation': op_id,
                'machine': machine,
                'start': start_time,
                'duration': proc_time
            })
            
            # Apply action
            state = self._apply_action(state, action)
        
        # Compute final makespan
        makespan = np.max(state['machine_available'])
        
        return {
            'makespan': makespan,
            'schedule': schedule,
            'n_decisions': len(schedule)
        }
