"""
Gurobi MIP expert for optimal job-shop scheduling solutions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class GurobiExpert:
    """
    Uses Gurobi MIP solver to find optimal or near-optimal solutions
    for job-shop scheduling problems.
    """
    
    def __init__(self, instance, time_limit: int = 300, gap_limit: float = 0.01):
        """
        Initialize Gurobi expert.
        
        Args:
            instance: JSPInstance object
            time_limit: Maximum solving time in seconds
            gap_limit: MIP gap limit for optimality
        """
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is not available. Please install gurobipy.")
        
        self.instance = instance
        self.time_limit = time_limit
        self.gap_limit = gap_limit
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
    
    def solve(self) -> Dict:
        """
        Solve the JSP instance using Gurobi MIP.
        
        Returns:
            Dictionary containing:
                - 'makespan': Optimal/near-optimal makespan
                - 'schedule': Job operation start times
                - 'sequence': Operation sequence on each machine
                - 'status': Solver status
        """
        try:
            model = gp.Model("JSP")
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.gap_limit)
            model.setParam('OutputFlag', 0)  # Suppress output
            
            # Decision variables
            # Start time of operation (job, op_id)
            start = {}
            for j in range(self.n_jobs):
                for o in range(self.n_machines):
                    start[j, o] = model.addVar(vtype=GRB.CONTINUOUS, 
                                               name=f"start_{j}_{o}")
            
            # Makespan variable
            makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
            
            # Big M for disjunctive constraints
            M = sum(self.instance.processing_times.flatten())
            
            # Binary variables for operation sequencing on machines
            y = {}
            for m in range(self.n_machines):
                ops_on_machine = []
                for j in range(self.n_jobs):
                    for o in range(self.n_machines):
                        if self.instance.machine_order[j, o] == m:
                            ops_on_machine.append((j, o))
                
                for i in range(len(ops_on_machine)):
                    for k in range(i + 1, len(ops_on_machine)):
                        j1, o1 = ops_on_machine[i]
                        j2, o2 = ops_on_machine[k]
                        y[j1, o1, j2, o2] = model.addVar(vtype=GRB.BINARY,
                                                         name=f"y_{j1}_{o1}_{j2}_{o2}")
            
            model.update()
            
            # Objective: minimize makespan
            model.setObjective(makespan, GRB.MINIMIZE)
            
            # Precedence constraints within jobs
            for j in range(self.n_jobs):
                for o in range(self.n_machines - 1):
                    _, pt = self.instance.get_operation(j, o)
                    model.addConstr(start[j, o] + pt <= start[j, o + 1])
            
            # Makespan constraints
            for j in range(self.n_jobs):
                o = self.n_machines - 1
                _, pt = self.instance.get_operation(j, o)
                model.addConstr(start[j, o] + pt <= makespan)
            
            # Machine capacity constraints (disjunctive)
            for m in range(self.n_machines):
                ops_on_machine = []
                for j in range(self.n_jobs):
                    for o in range(self.n_machines):
                        if self.instance.machine_order[j, o] == m:
                            ops_on_machine.append((j, o))
                
                for i in range(len(ops_on_machine)):
                    for k in range(i + 1, len(ops_on_machine)):
                        j1, o1 = ops_on_machine[i]
                        j2, o2 = ops_on_machine[k]
                        _, pt1 = self.instance.get_operation(j1, o1)
                        _, pt2 = self.instance.get_operation(j2, o2)
                        
                        # Either op1 before op2 or op2 before op1
                        model.addConstr(start[j1, o1] + pt1 <= start[j2, o2] + 
                                      M * (1 - y[j1, o1, j2, o2]))
                        model.addConstr(start[j2, o2] + pt2 <= start[j1, o1] + 
                                      M * y[j1, o1, j2, o2])
            
            # Solve
            model.optimize()
            
            # Extract solution
            result = {
                'makespan': makespan.X if model.status == GRB.OPTIMAL else None,
                'schedule': {},
                'sequence': {m: [] for m in range(self.n_machines)},
                'status': model.status
            }
            
            if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                for j in range(self.n_jobs):
                    for o in range(self.n_machines):
                        result['schedule'][j, o] = start[j, o].X
                        machine, _ = self.instance.get_operation(j, o)
                        result['sequence'][machine].append((j, o, start[j, o].X))
                
                # Sort sequences by start time
                for m in range(self.n_machines):
                    result['sequence'][m].sort(key=lambda x: x[2])
            
            return result
            
        except Exception as e:
            print(f"Gurobi solve error: {e}")
            return {
                'makespan': None,
                'schedule': {},
                'sequence': {},
                'status': 'ERROR'
            }
    
    def get_expert_action(self, state: Dict) -> int:
        """
        Get expert's action (job selection) for a given state.
        
        This is a simplified version that uses the full solution
        to determine which job should be scheduled next.
        
        Args:
            state: Current scheduling state
            
        Returns:
            Job ID to schedule next
        """
        solution = self.solve()
        
        if solution['status'] not in [GRB.OPTIMAL, 2]:  # 2 is SUBOPTIMAL
            # Fallback to simple heuristic
            return self._fallback_heuristic(state)
        
        current_time = state.get('current_time', 0)
        job_progress = state.get('job_progress', np.zeros(self.n_jobs, dtype=int))
        available_jobs = state.get('available_jobs', list(range(self.n_jobs)))
        
        # Find the job whose next operation should start earliest according to expert
        best_job = None
        earliest_time = float('inf')
        
        for job in available_jobs:
            op_id = job_progress[job]
            if op_id < self.n_machines:
                if (job, op_id) in solution['schedule']:
                    start_time = solution['schedule'][job, op_id]
                    if start_time < earliest_time:
                        earliest_time = start_time
                        best_job = job
        
        return best_job if best_job is not None else available_jobs[0]
    
    def _fallback_heuristic(self, state: Dict) -> int:
        """Simple SPT (Shortest Processing Time) heuristic as fallback."""
        job_progress = state.get('job_progress', np.zeros(self.n_jobs, dtype=int))
        available_jobs = state.get('available_jobs', list(range(self.n_jobs)))
        
        min_time = float('inf')
        best_job = available_jobs[0]
        
        for job in available_jobs:
            op_id = job_progress[job]
            if op_id < self.n_machines:
                _, proc_time = self.instance.get_operation(job, op_id)
                if proc_time < min_time:
                    min_time = proc_time
                    best_job = job
        
        return best_job
