#!/usr/bin/env python3
"""
Setup script for creating necessary directories and downloading OR-Library instances.
"""

import os
import sys
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_directories(config):
    """Create necessary directories."""
    directories = [
        config['instance']['data_dir'],
        config['output']['results_dir'],
        config['output']['models_dir'],
        config['output']['logs_dir'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def create_sample_instance(data_dir):
    """Create a sample 10x10 JSP instance in OR-Library format."""
    filepath = os.path.join(data_dir, 'sample_10x10.txt')
    
    # Sample 10x10 instance (randomly generated)
    with open(filepath, 'w') as f:
        f.write("10 10\n")
        # Each line: machine_0 time_0 machine_1 time_1 ... machine_9 time_9
        sample_data = [
            "0 29 1 78 2 9 3 36 4 49 5 11 6 62 7 56 8 44 9 21",
            "0 43 2 90 4 75 9 11 3 69 1 28 6 46 5 46 7 72 8 30",
            "1 91 0 85 3 39 2 74 8 90 5 10 7 12 6 89 9 45 4 33",
            "1 81 2 95 0 71 4 99 6 9 8 52 7 85 3 98 9 22 5 43",
            "2 14 0 6 1 22 5 61 3 26 4 69 8 21 7 49 9 72 6 53",
            "2 84 1 2 5 52 3 95 8 48 9 72 0 47 6 65 4 6 7 25",
            "0 46 1 37 2 61 3 13 6 32 5 21 9 32 8 89 7 30 4 55",
            "1 31 0 86 3 46 5 74 4 32 6 88 9 19 8 48 7 36 2 79",
            "0 76 2 69 6 76 5 51 1 85 9 11 3 40 7 89 4 26 8 74",
            "1 85 0 13 2 61 6 7 8 64 9 76 5 47 3 52 4 90 7 45"
        ]
        
        for line in sample_data:
            f.write(line + "\n")
    
    print(f"Created sample instance: {filepath}")


def create_readme():
    """Create a simple README for the data directory."""
    readme_content = """# pilotJSP Data Directory

This directory contains:
- JSP instance files in OR-Library format
- Generated instances for training and testing
- CSV exports of instance data

## OR-Library Format

Each instance file has the format:
```
n_jobs n_machines
job_0: machine_0 time_0 machine_1 time_1 ... machine_n time_n
job_1: machine_0 time_0 machine_1 time_1 ... machine_n time_n
...
```

## Files

- `sample_10x10.txt`: Sample 10x10 JSP instance
- `*.csv`: CSV exports of instances for data analysis
"""
    
    with open('data/README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created data/README.md")


def main():
    """Main setup function."""
    print("=" * 60)
    print("pilotJSP Setup")
    print("=" * 60)
    
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_path = os.path.join(project_dir, 'config.yaml')
    
    # Change to project directory
    os.chdir(project_dir)
    
    config = load_config(config_path)
    
    print("\n1. Creating directories...")
    create_directories(config)
    
    print("\n2. Creating sample instance...")
    create_sample_instance(config['instance']['data_dir'])
    
    print("\n3. Creating documentation...")
    create_readme()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run example: python scripts/run_example.py")
    print("3. Explore the pilotjsp package and customize for your needs")


if __name__ == '__main__':
    main()
