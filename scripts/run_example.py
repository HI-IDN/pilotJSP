#!/usr/bin/env python3
"""
Example script demonstrating the pilotJSP workflow.

This script shows how to:
1. Generate/load JSP instances
2. Extract features
3. Query expert solver
4. Build preference pairs
5. Train with DAgger
6. Evaluate pilot heuristic
"""

import os
import sys
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilotjsp import (
    JSPInstance,
    FeatureExtractor,
    GurobiExpert,
    PreferenceBuilder,
    DAggerAlgorithm,
    OrdinalRegressionModel,
    PilotHeuristic
)
from pilotjsp.utils import ensure_dir, save_results, print_statistics


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_instances(config, n_instances=10):
    """Generate random JSP instances."""
    instances = []
    
    for i in range(n_instances):
        instance = JSPInstance.generate_random(
            n_jobs=config['instance']['n_jobs'],
            n_machines=config['instance']['n_machines'],
            seed=config['instance']['seed'] + i
        )
        instances.append(instance)
    
    return instances


def simple_environment(instance):
    """
    Simple environment wrapper for DAgger.
    """
    class SimpleJSPEnv:
        def __init__(self, instance):
            self.instance = instance
            self.reset()
        
        def reset(self):
            self.state = {
                'job_progress': np.zeros(instance.n_jobs, dtype=int),
                'current_time': 0,
                'machine_available': np.zeros(instance.n_machines),
                'available_jobs': list(range(instance.n_jobs)),
                'job_release': np.zeros(instance.n_jobs)
            }
            return self.state
        
        def step(self, action):
            # Apply action (simplified)
            job_progress = self.state['job_progress']
            op_id = job_progress[action]
            
            if op_id >= instance.n_machines:
                # Job complete
                return self.state, 0, True, {}
            
            machine, proc_time = instance.get_operation(action, op_id)
            
            start_time = max(self.state['current_time'], 
                           self.state['machine_available'][machine])
            completion_time = start_time + proc_time
            
            self.state['machine_available'][machine] = completion_time
            job_progress[action] += 1
            self.state['current_time'] = completion_time
            
            # Update available jobs
            available = []
            for j in range(instance.n_jobs):
                if job_progress[j] < instance.n_machines:
                    available.append(j)
            self.state['available_jobs'] = available
            
            done = len(available) == 0
            reward = -completion_time * 0.01
            
            return self.state, reward, done, {}
    
    return SimpleJSPEnv(instance)


def main():
    """Main execution function."""
    print("=" * 60)
    print("pilotJSP: Imitation Learning for Job-Shop Scheduling")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)
    
    # Create output directories
    ensure_dir(config['output']['results_dir'])
    ensure_dir(config['output']['models_dir'])
    ensure_dir(config['output']['logs_dir'])
    ensure_dir(config['instance']['data_dir'])
    
    print("\n1. Generating JSP instances...")
    instances = generate_instances(config, n_instances=5)
    print(f"   Generated {len(instances)} instances")
    
    # Save first instance as example
    instance_path = os.path.join(config['instance']['data_dir'], 'example_10x10.txt')
    instances[0].to_orlibrary_format(instance_path)
    print(f"   Saved example instance to {instance_path}")
    
    # Save to CSV format
    csv_path = os.path.join(config['instance']['data_dir'], 'example_10x10.csv')
    instances[0].to_csv(csv_path)
    print(f"   Saved example instance to CSV: {csv_path}")
    
    print("\n2. Extracting features...")
    feature_extractor = FeatureExtractor(instances[0])
    
    # Example state
    example_state = {
        'job_progress': np.zeros(config['instance']['n_jobs'], dtype=int),
        'current_time': 0,
        'machine_available': np.zeros(config['instance']['n_machines']),
        'job_release': np.zeros(config['instance']['n_jobs']),
        'candidate_job': 0
    }
    
    features = feature_extractor.extract_features(example_state)
    print(f"   Extracted {len(features)} features")
    print(f"   Feature names: {feature_extractor.FEATURE_NAMES[:5]}... (showing first 5)")
    
    # Save features example
    features_path = os.path.join(config['output']['results_dir'], 'example_features.csv')
    feature_extractor.save_features_to_csv(features.reshape(1, -1), features_path)
    print(f"   Saved example features to {features_path}")
    
    print("\n3. Initializing Gurobi expert...")
    try:
        expert = GurobiExpert(
            instances[0],
            time_limit=config['expert']['time_limit'],
            gap_limit=config['expert']['gap_limit']
        )
        print("   Gurobi expert initialized successfully")
        print("   Note: Actual solving requires Gurobi license")
    except ImportError:
        print("   Warning: Gurobi not available, using fallback heuristics")
        expert = None
    
    print("\n4. Building preference pairs...")
    if expert is not None:
        preference_builder = PreferenceBuilder(feature_extractor, expert)
        
        # Create some example states
        example_states = [example_state]
        example_actions = [0]
        
        pairs = preference_builder.build_pairs_from_trajectory(
            example_states, example_actions
        )
        print(f"   Built {len(pairs)} preference pairs")
        
        # Save pairs
        pairs_path = os.path.join(config['output']['results_dir'], 'preference_pairs.csv')
        preference_builder.add_pairs(pairs)
        preference_builder.save_to_csv(pairs_path)
        print(f"   Saved preference pairs to {pairs_path}")
    else:
        print("   Skipping (requires Gurobi)")
    
    print("\n5. Training ordinal regression model...")
    model = OrdinalRegressionModel(
        n_features=config['features']['n_features'],
        learning_rate=config['model']['learning_rate'],
        n_epochs=config['model']['n_epochs'],
        regularization=config['model']['regularization']
    )
    
    # Train on dummy data for demonstration
    n_samples = 100
    features1 = np.random.randn(n_samples, 13)
    features2 = np.random.randn(n_samples, 13)
    labels = np.random.choice([-1, 1], n_samples)
    
    model.fit(features1, features2, labels, verbose=False)
    accuracy = model.score(features1, features2, labels)
    print(f"   Model trained, accuracy: {accuracy:.3f}")
    
    # Save model
    model_path = os.path.join(config['output']['models_dir'], 'ordinal_model.pkl')
    model.save(model_path)
    print(f"   Saved model to {model_path}")
    
    # Print feature importance
    importance = model.get_feature_importance()
    print(f"   Top 3 important features: {np.argsort(-importance)[:3]}")
    
    print("\n6. Evaluating pilot heuristic...")
    pilot = PilotHeuristic(
        instances[0],
        feature_extractor,
        model,
        rollout_depth=config['pilot']['rollout_depth']
    )
    
    solution = pilot.solve()
    print(f"   Pilot heuristic makespan: {solution['makespan']:.1f}")
    print(f"   Number of decisions: {solution['n_decisions']}")
    
    # Save solution
    solution_path = os.path.join(config['output']['results_dir'], 'pilot_solution.json')
    save_results(solution, solution_path, format='json')
    print(f"   Saved solution to {solution_path}")
    
    print("\n7. Summary:")
    print(f"   - Generated {len(instances)} JSP instances")
    print(f"   - Extracted {config['features']['n_features']} features per state")
    print(f"   - Trained ordinal regression model")
    print(f"   - Evaluated pilot heuristic")
    print(f"   - Results saved to {config['output']['results_dir']}/")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
