#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram



@dataclass
class KnapsackProblem:
    values: list
    weights: list
    max_weight: int
    total_weight: int = field(init=False)
        
    def __post_init__(self):
        self.total_weight = sum(self.weights)


##############################################################################
# Soft Constraint: Quadratic Penalty
##############################################################################


class QuadPhaseCirc(QuantumCircuit):
    """
    Phase seperation circuit for Knapsack QAOA with quadratic soft constraints.
    """
    
    def __init__(self, problem: KnapsackProblem):
        # create registers
        nx, ny = QuadQAOACirc.register_sizes(problem)
        qx = QuantumRegister(nx)
        qy = QuantumRegister(ny)
        # create circuit
        super().__init__(qx, qy, name="UPhase")
        
        # parameters for this circuit
        self.gamma = Parameter("gamma")
        self.A = Parameter("A")
        self.B = Parameter("B")

        # Single-qubit rotations on x register
        for idx, (value, weight) in enumerate(zip(problem.values, problem.weights)):
            angle = (self.A / 2 * (problem.total_weight - (problem.max_weight**2 - problem.max_weight) / 2 * weight)
                        - self.B * value / 2) * self.gamma
            super().rz(2 * angle, qx[idx])

        # Single-qubit rotations on y register
        for idx in range(ny):
            angle = (problem.max_weight / 2 - 1
                        + (idx+1) * ((problem.max_weight**2 + problem.max_weight) / 4 - problem.total_weight / 2)) * self.gamma
            super().rz(2 * angle, qy[idx])

        super().barrier()

        # Two-qubit rotations on x register
        for idx1, weight1 in enumerate(problem.weights):
            for idx2, weight2 in enumerate(problem.weights[:idx1]):
                angle = weight1 * weight2 / 2 * self.gamma
                super().rzz(2 * angle, qx[idx1], qx[idx2])

        # Two-qubit rotations on y register
        for idx1 in range(ny):
            for idx2 in range(idx1):
                angle = (1 + (idx1 + 1) * (idx2 + 1)) / 2 * self.gamma
                super().rzz(2 * angle, qy[idx1], qy[idx2])

        super().barrier()

        # Common x and y register rotations
        for (ix, weight), iy in product(enumerate(problem.weights), range(ny)):
            angle = - (iy + 1) * weight / 2 * self.gamma
            super().rzz(2 * angle, qx[ix], qy[iy])


class QuadMixCirc(QuantumCircuit):
    """
    Mixer circuit for Knapsack QAOA with quadratic soft constraints.
    """
    
    def __init__(self, problem: KnapsackProblem):
        nx, ny = QuadQAOACirc.register_sizes(problem)
        # create registers
        q = QuantumRegister(nx+ny)
        # create circuit
        super().__init__(q, name="UMix")
        
        # parameters for this circuit
        self.beta = Parameter("beta")
        
        # x-rotation of all qubits
        super().rx(2 * self.beta, q)

    
class QuadQAOACirc(QuantumCircuit):
    """
    QAOA Circuit for Knapsack Problem with quadratic soft constraints.
    """
    
    def __init__(self, problem: KnapsackProblem, p: int):
        self.nx, self.ny = QuadQAOACirc.register_sizes(problem)
        qx = QuantumRegister(self.nx, name="qx")
        qy = QuantumRegister(self.ny, name="qy")
        qreg = (*qx, *qy)

        # Phase seperation circuit
        phase_circ = QuadPhaseCirc(problem)
        # Mixing circuit
        mix_circ = QuadMixCirc(problem)
        
        # Implementation of QAOA Circuit
        self.p = p

        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        self.A = Parameter("A")
        self.B = Parameter("B")

        super().__init__(qx, qy)

        # initialize
        super().h(qreg)

        # alternating application of phase seperation and mixing unitaries
        for gamma, beta in zip(self.gammas, self.betas):
            # application of phase seperation unitary
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.A: self.A,
                phase_circ.B: self.B,
            }
            phase_instruction = phase_circ.to_instruction(phase_params)
            super().append(phase_instruction, qreg)
            
            # application of mixing unitary
            mix_params = {
                mix_circ.beta: beta,
            }
            mix_instruction = mix_circ.to_instruction(mix_params)
            super().append(mix_instruction, qreg)

        # measurement
        super().measure_all()
        
    @staticmethod
    def register_sizes(problem: KnapsackProblem):
        # number of bits for x register
        nx = len(problem.values)
        # number of bits for y register
        ny = problem.max_weight
        return nx, ny


class QuadQAOA():
    def __init__(self, problem: KnapsackProblem, p: int):
        self.problem = problem
        self.circuit = QuadQAOACirc(problem, p)
        
    def objective_func(self, bitstring: str, A: float, B: float):
        """
        Computes an objective function for the knapsack problem with quadratic soft constraints.
        """
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        xbits = np.array(bits[:self.circuit.nx])
        ybits = np.array(bits[self.circuit.nx:])
        penalty = (A * (1 - sum(ybits))**2
                    + A * (np.arange(1, self.circuit.ny+1).dot(ybits) - xbits.dot(self.problem.weights))**2)
        value = B * xbits.dot(self.problem.values)
        return penalty - value
    
    def counts_to_choices(self, counts):
        choices = {}
        for bitstring, count in counts.items():
            choice = bitstring[self.circuit.ny:]
            if not choice in choices.keys():
                choices[choice] = count
            else:
                choices[choice] += count
        return choices


##############################################################################
# Soft Constraint: Linear Penalty
##############################################################################


##############################################################################
# Hard Constraint: Random Walk
##############################################################################


##############################################################################
# Hard Constraint: Heuristic Mixer
##############################################################################


##############################################################################
# Functions
##############################################################################

def average(counts: dict, objective_func):
    """
    Computes average value of objective function for a given set of
    measurement results.
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = objective_func(bitstring)
        avg += obj * count
        sum_count += count
    return avg/sum_count

def simulate_circuit(parameter_values, transpiled_circuit: QuantumCircuit,
                     variable_parameters: list, fixed_parameters: dict = None,
                     backend = None, shots = 512, **kwargs):
    # ensure this is a valid dict
    if fixed_parameters is None:
        fixed_parameters = {}
    # create a parameter dictionary
    parameter_dict = {
        **dict(zip(variable_parameters, parameter_values)),
        **fixed_parameters,
    }
    # bind parameters of the circuit
    bound_circuit = transpiled_circuit.bind_parameters(parameter_dict)

    # run simulation
    job = execute(bound_circuit, backend, shots=shots, **kwargs)
    result = job.result()
    counts = result.get_counts()
    return counts

def get_noise_params(noise_backend):
    # Get necessary data for noise simulation from the backend
    noise_model = NoiseModel.from_backend(noise_backend)
    coupling_map = noise_backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    
    # create parameter dict
    noise_params = {
        "noise_model": noise_model,
        "coupling_map": coupling_map,
        "basis_gates": basis_gates,
    }
    
    return noise_params

##############################################################################
# Examples
##############################################################################

def example1():
    # define problem
    simple_problem = KnapsackProblem(values=[1, 2], weights=[2, 2],
                                     max_weight=2)
    # create QAOA circuit
    qaoa = QuadQAOA(simple_problem, p=1)
    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(qaoa.circuit, backend)
    sim = partial(simulate_circuit,
                  transpiled_circuit = transpiled_circuit,
                  variable_parameters = [*qaoa.circuit.betas,
                                         *qaoa.circuit.gammas,
                                         qaoa.circuit.A,
                                         qaoa.circuit.B],
                  backend = backend)
    param_vals = np.array([2.20803269, 1.66682401, 2.7, 1.1])
    counts = sim(param_vals)
    choices = qaoa.counts_to_choices(counts)
    plot_histogram(choices)
    plt.show()
    
def example2():
    # define problem
    simple_problem = KnapsackProblem(values=[1, 2], weights=[2, 2],
                                     max_weight=2)
    qaoa = QuadQAOA(simple_problem, p=1)
    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(qaoa.circuit, backend)
    A = 2.7
    B = 1.1
    sim = partial(simulate_circuit,
                  transpiled_circuit = transpiled_circuit,
                  variable_parameters = [*qaoa.circuit.betas,
                                         *qaoa.circuit.gammas],
                  fixed_parameters = {qaoa.circuit.A: A, qaoa.circuit.B: B},
                  backend = backend)
    objective_func = partial(qaoa.objective_func, A=A, B=B)
    def evaluate(vals):
        counts = sim(vals)
        avg = average(counts, objective_func)
        return avg

    n = 100
    betas = np.linspace(-np.pi/2, np.pi/2, n)
    gammas = np.linspace(-3*np.pi, 3*np.pi, n)
    valarr = np.array(list(product(betas, gammas)))
    avgs = [evaluate(vals) for vals in valarr]
    reshaped = np.transpose(np.reshape(avgs, (n, n)))
    plt.imshow(reshaped, interpolation="bilinear", origin="lower")
    plt.show()

def noise_example():
    # Get credentials for IBMQ
    provider = IBMQ.load_account()
    # Get IBM Quito as backend to base our noise simulation on
    noise_backend = provider.get_backend('ibmq_quito')
    # Get necessary parameters for noise simulation
    noise_params = get_noise_params(noise_backend)

    # define problem
    simple_problem = KnapsackProblem(values=[1, 2], weights=[2, 2],
                                     max_weight=2)
    # create QAOA object for this problem
    qaoa = QuadQAOA(simple_problem, p=1)
    # Use the Aer simulator as backend for simulation
    sim_backend = Aer.get_backend("aer_simulator")
    # Transpile circuit to this backend
    transpiled_circuit = transpile(qaoa.circuit, sim_backend)
    
    # Define simulation function for this problem
    sim = partial(simulate_circuit,
                  transpiled_circuit = transpiled_circuit,
                  variable_parameters = [*qaoa.circuit.betas,
                                         *qaoa.circuit.gammas,
                                         qaoa.circuit.A,
                                         qaoa.circuit.B],
                  backend = sim_backend, **noise_params)
    
    # Define toy parameter values to use
    param_vals = np.array([2.20803269, 1.66682401, 2.7, 1.1])
    # Run the simulation
    counts = sim(param_vals)
    choices = qaoa.counts_to_choices(counts)
    plot_histogram(choices)
    plt.show()
    

def main():
    noise_example()

if __name__ == "__main__":
    main()
