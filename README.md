# pilotJSP

Code repository to accompany LION20 conference paper: Learning from Expert Optimization: Expert‑Like Lookahead Policies via Pilot Heuristics

## Overview

**pilotJSP** is a Python implementation of imitation learning techniques for job-shop scheduling problems (JSP). It combines:
- **Expert optimization** using Gurobi MIP solver
- **Feature extraction** with 13 dispatching rule features
- **DAgger algorithm** for dataset aggregation
- **Ordinal regression** for preference learning
- **Pilot heuristic** for lookahead-based scheduling

## Features

- 🏭 **OR-Library JSP Instances**: Generate and load 10x10 job-shop scheduling problems
- 📊 **Feature Extraction**: Extract 13 dispatching rule features from scheduling states
- 🧠 **Expert Solver**: Query Gurobi MIP solver for optimal/near-optimal solutions
- 📚 **Preference Learning**: Build preference pairs from expert demonstrations
- 🔄 **DAgger Algorithm**: Iterative dataset aggregation for imitation learning
- 📈 **Ordinal Regression**: Train models from preference pairs
- 🎯 **Pilot Heuristic**: Rollout-based lookahead for decision making
- 💾 **CSV Data Handling**: Easy data import/export

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HI-IDN/pilotJSP.git
cd pilotJSP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Gurobi for expert optimization:
   - Download from [gurobi.com](https://www.gurobi.com/)
   - Obtain a license (free academic licenses available)

## Quick Start

### 1. Setup

Run the setup script to create necessary directories and sample data:

```bash
python scripts/setup.py
```

### 2. Run Example

Execute the main example demonstrating the full workflow:

```bash
python scripts/run_example.py
```

This will:
- Generate JSP instances
- Extract features
- Initialize the expert solver
- Build preference pairs
- Train an ordinal regression model
- Evaluate the pilot heuristic

### 3. Evaluate Methods

Compare different dispatching rules and the pilot heuristic:

```bash
python scripts/evaluate.py
```

## Usage

### Generate JSP Instances

```python
from pilotjsp import JSPInstance

# Generate random 10x10 instance
instance = JSPInstance.generate_random(n_jobs=10, n_machines=10, seed=42)

# Save in OR-Library format
instance.to_orlibrary_format('data/my_instance.txt')

# Save as CSV
instance.to_csv('data/my_instance.csv')

# Load from file
instance = JSPInstance.from_orlibrary('my_instance.txt', data_dir='data')
```

### Extract Features

```python
from pilotjsp import FeatureExtractor

feature_extractor = FeatureExtractor(instance)

# Define a state
state = {
    'job_progress': [0, 0, 0, ...],
    'current_time': 0,
    'machine_available': [0, 0, 0, ...],
    'candidate_job': 0
}

# Extract 13 features
features = feature_extractor.extract_features(state)
```

### Train with DAgger

```python
from pilotjsp import (
    GurobiExpert, 
    PreferenceBuilder, 
    DAggerAlgorithm,
    OrdinalRegressionModel
)

# Initialize components
expert = GurobiExpert(instance, time_limit=300)
feature_extractor = FeatureExtractor(instance)
preference_builder = PreferenceBuilder(feature_extractor, expert)
model = OrdinalRegressionModel(n_features=13)

# Run DAgger
dagger = DAggerAlgorithm(expert, feature_extractor, preference_builder, 
                         model, n_iterations=10)
results = dagger.run(env, n_episodes_per_iter=10)
```

### Evaluate Pilot Heuristic

```python
from pilotjsp import PilotHeuristic

pilot = PilotHeuristic(instance, feature_extractor, model, rollout_depth=5)
solution = pilot.solve()

print(f"Makespan: {solution['makespan']}")
print(f"Schedule: {solution['schedule']}")
```

## Configuration

Edit `config.yaml` to customize:
- Instance parameters (size, seed)
- Expert solver settings (time limit, gap)
- DAgger hyperparameters (iterations, beta decay)
- Model parameters (learning rate, epochs)
- Pilot heuristic settings (rollout depth)

## Project Structure

```
pilotJSP/
├── pilotjsp/              # Main package
│   ├── __init__.py        # Package initialization
│   ├── jsp_instance.py    # JSP instance handling
│   ├── features.py        # Feature extraction
│   ├── expert.py          # Gurobi MIP expert
│   ├── preferences.py     # Preference pair builder
│   ├── dagger.py          # DAgger algorithm
│   ├── model.py           # Ordinal regression model
│   ├── pilot.py           # Pilot heuristic
│   └── utils.py           # Utility functions
├── scripts/               # Example scripts
│   ├── setup.py           # Setup script
│   ├── run_example.py     # Main example
│   └── evaluate.py        # Evaluation script
├── data/                  # Data directory
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 13 Dispatching Rule Features

The feature extractor computes these features for each scheduling decision:

1. **Processing Time (PT)**: Duration of current operation
2. **Remaining Processing Time (RPT)**: Sum of remaining operations
3. **Number of Remaining Operations (NRO)**: Count of pending operations
4. **Work Remaining (WR)**: Total work left for the job
5. **Slack Time (ST)**: Time buffer before due date
6. **Critical Ratio (CR)**: Ratio of remaining time to remaining work
7. **Shortest Processing Time (SPT)**: For priority ranking
8. **Earliest Due Date (EDD)**: Estimated due date
9. **Arrival Time (FCFS)**: Job release time
10. **Machine Workload (MW)**: Current machine utilization
11. **Machine Waiting Time (MWT)**: Idle time on machine
12. **Operation Due Date (ODD)**: Estimated operation deadline
13. **Total Work Content (TWC)**: Complete job processing time

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{pilotjsp2020,
  title={Learning from Expert Optimization: Expert-Like Lookahead Policies via Pilot Heuristics},
  author={Your Name},
  booktitle={LION20 Conference},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OR-Library for benchmark instances
- Gurobi Optimization for the MIP solver
- The DAgger algorithm from Ross et al. (2011)
