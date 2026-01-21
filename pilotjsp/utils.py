"""
Utility functions for CSV data handling and general operations.
"""

import csv
import json
import os
from typing import List, Dict, Any
import numpy as np


def save_dict_to_csv(data: Dict[str, List], filepath: str):
    """
    Save a dictionary of lists to CSV file.
    
    Args:
        data: Dictionary with column names as keys and data lists as values
        filepath: Path to save CSV file
    """
    if not data:
        return
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        
        # Transpose data
        n_rows = len(next(iter(data.values())))
        for i in range(n_rows):
            row = {key: data[key][i] for key in data.keys()}
            writer.writerow(row)


def load_csv_to_dict(filepath: str) -> Dict[str, List]:
    """
    Load CSV file to dictionary of lists.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary with column names as keys and data lists as values
    """
    data = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                
                # Try to convert to number
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    
    return data


def save_results(results: Dict, filepath: str, format: str = 'json'):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
        format: File format ('json' or 'csv')
    """
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            elif isinstance(value, list):
                # Handle lists that may contain numpy types
                serializable_results[key] = [
                    item.item() if isinstance(item, (np.integer, np.floating))
                    else convert_numpy_in_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            elif isinstance(value, dict):
                serializable_results[key] = convert_numpy_in_dict(value)
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == 'csv':
        # Flatten results for CSV
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                for i, item in enumerate(value):
                    flat_results[f"{key}_{i}"] = [item]
            else:
                flat_results[key] = [value]
        
        save_dict_to_csv(flat_results, filepath)


def convert_numpy_in_dict(d: Dict) -> Dict:
    """
    Recursively convert numpy types in dictionary to Python types.
    
    Args:
        d: Dictionary to convert
        
    Returns:
        Dictionary with numpy types converted
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = value.item()
        elif isinstance(value, dict):
            result[key] = convert_numpy_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                item.item() if isinstance(item, (np.integer, np.floating))
                else convert_numpy_in_dict(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def load_results(filepath: str) -> Dict:
    """
    Load results from file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Results dictionary
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif ext == '.csv':
        return load_csv_to_dict(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with statistics
    """
    values = np.array(values)
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'count': len(values)
    }


def print_statistics(name: str, values: List[float]):
    """
    Print statistics for a list of values.
    
    Args:
        name: Name of the metric
        values: List of numerical values
    """
    stats = compute_statistics(values)
    
    print(f"\n{name} Statistics:")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Count:  {stats['count']}")


class CSVLogger:
    """
    Logger for writing results to CSV incrementally.
    """
    
    def __init__(self, filepath: str, fieldnames: List[str]):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file
            fieldnames: List of field names
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
    
    def log(self, data: Dict):
        """
        Log a row of data.
        
        Args:
            data: Dictionary with data to log
        """
        self.writer.writerow(data)
        self.file.flush()
    
    def close(self):
        """Close the CSV file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
