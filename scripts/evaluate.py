#!/usr/bin/env python3
"""
Script to evaluate different dispatching rules and the learned pilot heuristic.
"""

import os
import sys
import yaml
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilotjsp import JSPInstance, FeatureExtractor, OrdinalRegressionModel, PilotHeuristic
from pilotjsp.utils import ensure_dir, save_results, print_statistics


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SimpleHeuristic:
    """Base class for simple dispatching rules."""
    
    def __init__(self, instance, rule='SPT'):
        self.instance = instance
        self.rule = rule
    
    def solve(self, initial_state=None):
        """Solve using simple dispatching rule."""
        if initial_state is None:
            state = {
                'job_progress': np.zeros(self.instance.n_jobs, dtype=int),
                'current_time': 0,
                'machine_available': np.zeros(self.instance.n_machines),
                'available_jobs': list(range(self.instance.n_jobs)),
            }
        else:
            state = initial_state.copy()
        
        schedule = []
        
        while len(state['available_jobs']) > 0:
            # Select job based on rule
            if self.rule == 'SPT':
                job = self._spt_rule(state)
            elif self.rule == 'LPT':
                job = self._lpt_rule(state)
            elif self.rule == 'FCFS':
                job = state['available_jobs'][0]
            elif self.rule == 'random':
                job = np.random.choice(state['available_jobs'])
            else:
                job = state['available_jobs'][0]
            
            # Apply action
            op_id = state['job_progress'][job]
            machine, proc_time = self.instance.get_operation(job, op_id)
            
            start_time = max(state['current_time'], state['machine_available'][machine])
            
            schedule.append({
                'job': job,
                'operation': op_id,
                'machine': machine,
                'start': start_time,
                'duration': proc_time
            })
            
            # Update state
            state['machine_available'][machine] = start_time + proc_time
            state['job_progress'][job] += 1
            state['current_time'] = start_time + proc_time
            
            # Update available jobs
            available = []
            for j in range(self.instance.n_jobs):
                if state['job_progress'][j] < self.instance.n_machines:
                    available.append(j)
            state['available_jobs'] = available
        
        makespan = np.max(state['machine_available'])
        
        return {
            'makespan': makespan,
            'schedule': schedule,
            'n_decisions': len(schedule)
        }
    
    def _spt_rule(self, state):
        """Shortest Processing Time rule."""
        best_job = state['available_jobs'][0]
        min_time = float('inf')
        
        for job in state['available_jobs']:
            op_id = state['job_progress'][job]
            _, proc_time = self.instance.get_operation(job, op_id)
            if proc_time < min_time:
                min_time = proc_time
                best_job = job
        
        return best_job
    
    def _lpt_rule(self, state):
        """Longest Processing Time rule."""
        best_job = state['available_jobs'][0]
        max_time = -1
        
        for job in state['available_jobs']:
            op_id = state['job_progress'][job]
            _, proc_time = self.instance.get_operation(job, op_id)
            if proc_time > max_time:
                max_time = proc_time
                best_job = job
        
        return best_job


def evaluate_methods(instances: List[JSPInstance], config: Dict) -> Dict:
    """Evaluate different methods on test instances."""
    results = {method: [] for method in config['evaluation']['baseline_methods']}
    results['pilot'] = []
    
    print("\nEvaluating methods on test instances...")
    
    for i, instance in enumerate(instances):
        print(f"\n  Instance {i+1}/{len(instances)}")
        
        # Evaluate baseline methods
        for method in config['evaluation']['baseline_methods']:
            heuristic = SimpleHeuristic(instance, rule=method)
            solution = heuristic.solve()
            makespan = solution['makespan']
            results[method].append(makespan)
            print(f"    {method:10s}: {makespan:6.1f}")
        
        # Evaluate pilot heuristic (with dummy model for now)
        feature_extractor = FeatureExtractor(instance)
        model = OrdinalRegressionModel(n_features=13)
        
        # Train on small random dataset
        features1 = np.random.randn(50, 13)
        features2 = np.random.randn(50, 13)
        labels = np.random.choice([-1, 1], 50)
        model.fit(features1, features2, labels, verbose=False)
        
        pilot = PilotHeuristic(instance, feature_extractor, model, 
                              rollout_depth=config['pilot']['rollout_depth'])
        solution = pilot.solve()
        makespan = solution['makespan']
        results['pilot'].append(makespan)
        print(f"    {'pilot':10s}: {makespan:6.1f}")
    
    return results


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("pilotJSP: Evaluation of Dispatching Rules and Pilot Heuristic")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)
    
    # Create output directory
    ensure_dir(config['output']['results_dir'])
    
    print("\n1. Generating test instances...")
    n_test = config['evaluation']['n_test_instances']
    instances = []
    
    for i in range(n_test):
        instance = JSPInstance.generate_random(
            n_jobs=config['instance']['n_jobs'],
            n_machines=config['instance']['n_machines'],
            seed=1000 + i
        )
        instances.append(instance)
    
    print(f"   Generated {len(instances)} test instances")
    
    print("\n2. Evaluating methods...")
    results = evaluate_methods(instances, config)
    
    print("\n3. Summary statistics:")
    for method in results.keys():
        print_statistics(method, results[method])
    
    # Save results
    results_path = os.path.join(config['output']['results_dir'], 'evaluation_results.json')
    save_results(results, results_path)
    print(f"\n   Results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
