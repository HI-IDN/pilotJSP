"""
Job-shop scheduling instance representation and OR-Library instance generation.
"""

import numpy as np
import os
from typing import List, Tuple, Optional
import csv


class JSPInstance:
    """
    Represents a job-shop scheduling instance.
    
    Attributes:
        n_jobs: Number of jobs
        n_machines: Number of machines
        processing_times: Processing times matrix [n_jobs x n_machines]
        machine_order: Machine order matrix [n_jobs x n_machines]
    """
    
    def __init__(self, n_jobs: int, n_machines: int):
        """
        Initialize a JSP instance.
        
        Args:
            n_jobs: Number of jobs
            n_machines: Number of machines
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.processing_times = np.zeros((n_jobs, n_machines), dtype=int)
        self.machine_order = np.zeros((n_jobs, n_machines), dtype=int)
    
    @classmethod
    def from_orlibrary(cls, instance_name: str, data_dir: str = "data") -> "JSPInstance":
        """
        Load a JSP instance from OR-Library format file.
        
        Args:
            instance_name: Name of the instance file
            data_dir: Directory containing instance files
            
        Returns:
            JSPInstance object
        """
        filepath = os.path.join(data_dir, instance_name)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # First line contains n_jobs and n_machines
        n_jobs, n_machines = map(int, lines[0].strip().split())
        instance = cls(n_jobs, n_machines)
        
        # Parse job data
        for i in range(n_jobs):
            line = lines[i + 1].strip().split()
            for j in range(n_machines):
                instance.machine_order[i, j] = int(line[j * 2])
                instance.processing_times[i, j] = int(line[j * 2 + 1])
        
        return instance
    
    @classmethod
    def generate_random(cls, n_jobs: int = 10, n_machines: int = 10, 
                       seed: Optional[int] = None) -> "JSPInstance":
        """
        Generate a random JSP instance similar to OR-Library 10x10 instances.
        
        Args:
            n_jobs: Number of jobs (default: 10)
            n_machines: Number of machines (default: 10)
            seed: Random seed for reproducibility
            
        Returns:
            JSPInstance object
        """
        if seed is not None:
            np.random.seed(seed)
        
        instance = cls(n_jobs, n_machines)
        
        # Generate processing times (typical range: 1-100)
        instance.processing_times = np.random.randint(1, 101, (n_jobs, n_machines))
        
        # Generate random machine orders (permutations)
        for i in range(n_jobs):
            instance.machine_order[i] = np.random.permutation(n_machines)
        
        return instance
    
    def to_orlibrary_format(self, filepath: str):
        """
        Save instance in OR-Library format.
        
        Args:
            filepath: Path to save the instance
        """
        with open(filepath, 'w') as f:
            f.write(f"{self.n_jobs} {self.n_machines}\n")
            
            for i in range(self.n_jobs):
                line = []
                for j in range(self.n_machines):
                    line.extend([
                        str(self.machine_order[i, j]),
                        str(self.processing_times[i, j])
                    ])
                f.write(" ".join(line) + "\n")
    
    def get_operation(self, job_id: int, op_id: int) -> Tuple[int, int]:
        """
        Get the machine and processing time for a specific operation.
        
        Args:
            job_id: Job index
            op_id: Operation index within the job
            
        Returns:
            Tuple of (machine_id, processing_time)
        """
        machine = self.machine_order[job_id, op_id]
        processing_time = self.processing_times[job_id, op_id]
        return machine, processing_time
    
    def to_csv(self, filepath: str):
        """
        Save instance data to CSV format.
        
        Args:
            filepath: Path to save the CSV file
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['job_id', 'operation_id', 'machine_id', 'processing_time'])
            
            for job_id in range(self.n_jobs):
                for op_id in range(self.n_machines):
                    machine, proc_time = self.get_operation(job_id, op_id)
                    writer.writerow([job_id, op_id, machine, proc_time])
    
    @classmethod
    def from_csv(cls, filepath: str) -> "JSPInstance":
        """
        Load instance from CSV format.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            JSPInstance object
        """
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        n_jobs = max(int(row['job_id']) for row in rows) + 1
        n_machines = max(int(row['operation_id']) for row in rows) + 1
        
        instance = cls(n_jobs, n_machines)
        
        for row in rows:
            job_id = int(row['job_id'])
            op_id = int(row['operation_id'])
            machine = int(row['machine_id'])
            proc_time = int(row['processing_time'])
            
            instance.machine_order[job_id, op_id] = machine
            instance.processing_times[job_id, op_id] = proc_time
        
        return instance
