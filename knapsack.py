#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from functools import partial
from itertools import product
import math

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram


##############################################################################
# General Auxillary Classes
##############################################################################


@dataclass
class KnapsackProblem:
    """
    Class for representing particular instances of the knapsack problem.
    
    Intended only for 0-1 integer knapsack problems, i.e. values and weights
    are supposed to be of same length and must contain integer values. While
    it is possible for this class to represent other versions of the knapsack
    problem, all the other code is written with this specific subtype of the
    problem in mind and will not work for non-integers.
    
    Attributes:
    values (list): the values of the items (use int values only!)
    weights (list): the weights of the items (use int weights only!)
    max_weight (int): the maximum weight (carry capacity) of the knapsack
    total_weight (int): the sum of all weights
    N (int): the number of items
    """
    
    values: list
    weights: list
    max_weight: int
        
    def __post_init__(self):
        self.total_weight = sum(self.weights)
        self.N = len(self.weights)


##############################################################################
# Auxillary Circuits
##############################################################################


class KAdder(QuantumCircuit):
    """
    Circuit for adding 2^k to a register.
    
    An implementation of an adding circuit described in de la Grand'rive and
    Hullo: "Knapsack Problem variants of QAOA for battery revenue
    optimisation" (2019).
    
    The circuit takes a quantum register of n qubits storing some binary
    number and adds 2^k to that number. For this a carry register of size n-1
    is required. This class also contains a controlled version of the
    circuit, where the state of a single control qubit decides whether an
    addition takes place or not, as well as a reversed version of the circuit
    for uncomputing.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    control: controls the addition (1 qubit, optional)
    main: the register to which the circuit adds (n qubits)
    carry: auxillary register necessary for addition (n-1 qubits)
    
    Note:
    The control qubit is only included as part of the registers if
    controlled == True. Otherwise this circuit is only defined on main and
    carry registers.
    """
    
    def __init__(self, n, k, controlled = False, uncompute = False):
        """
        Initialize the circuit.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        k (int): exponent of the number being added (the circuit adds 2^k)
        
        Keyword Arguments:
        controlled (bool): whether or not you want the controlled version of
            the circuit (default: False, i.e. no control qubit)
        uncompute (bool): whether or not you want the reversed version of the
            circuit for uncomputing (default: False, i.e. regular version of
            the circuit)
        
        Note:
        The control qubit is only included as part of the registers if
        controlled == True. Otherwise this circuit is only defined on main and
        carry registers.
        """
        # Create registers
        main = QuantumRegister(n, name="main")
        carry = QuantumRegister(n-1, name="carry")
        if controlled:
            control = QuantumRegister(1, name="control")
            registers = [control, main, carry]
        else:
            registers = [main, carry]
        
        # Create QuantumCircuit
        name = f"Uncompute Add {2**k}" if uncompute else f"Add {2**k}"
        super().__init__(*registers, name=name)
        
        # Build circuit
        if uncompute:
            self.build_uncompute(n, k, controlled, registers)
        else:
            self.build_regular(n, k, controlled, registers)
        
    def build_regular(self, n, k, controlled, registers):
        """
        Build the circuit in regular order.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        k (int): exponent of the number being added (the circuit adds 2^k)
        controlled (bool): whether or not you want the controlled version of
            the circuit (True means controlled version)
        registers (list): list of all the registers that the circuit acts on
        
        Note:
        The control qubit is only included as part of the registers if
        controlled == True. Otherwise this circuit is only defined on main and
        carry registers.
        """
        if controlled:
            control, main, carry = registers
        else:
            main, carry = registers
        
        # take care of special case
        if k == n-1:
            if controlled:
                super().cx(control, main[k])
            else:
                super().x(main[k])
            return
        
        if controlled:
            super().ccx(control, main[k], carry[k])
        else:
            super().cx(main[k], carry[k])
        
        for j in range(k+1, n-1):
            super().ccx(main[j], carry[j-1], carry[j])

        super().cx(carry[n-2], main[n-1])

        for j in reversed(range(k+1, n-1)):
            super().ccx(main[j], carry[j-1], carry[j])
            super().cx(carry[j-1], main[j])
        
        if controlled:
            super().ccx(control, main[k], carry[k])
            super().cx(control, main[k])
        else:
            super().cx(main[k], carry[k])
            super().x(main[k])
    
    def build_uncompute(self, n, k, controlled, registers):
        """
        Build the circuit in reverse order for uncomputing.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        k (int): exponent of the number being added (the circuit adds 2^k)
        controlled (bool): whether or not you want the controlled version of
            the circuit (True means controlled version)
        registers (list): list of all the registers that the circuit acts on
        
        Note:
        The control qubit is only included as part of the registers if
        controlled == True. Otherwise this circuit is only defined on main and
        carry registers.
        """
        if controlled:
            control, main, carry = registers
        else:
            main, carry = registers
        
        # take care of special case
        if k == n-1:
            if controlled:
                super().cx(control, main[k])
            else:
                super().x(main[k])
            return
        
        if controlled:
            super().cx(control, main[k])
            super().ccx(control, main[k], carry[k])
        else:
            super().x(main[k])
            super().cx(main[k], carry[k])
        
        for j in range(k+1, n-1):
            super().cx(carry[j-1], main[j])
            super().ccx(main[j], carry[j-1], carry[j])

        super().cx(carry[n-2], main[n-1])

        for j in reversed(range(k+1, n-1)):
            super().ccx(main[j], carry[j-1], carry[j])

        if controlled:
            super().ccx(control, main[k], carry[k])
        else:
            super().cx(main[k], carry[k])


class Adder(QuantumCircuit):
    """
    Circuit for adding an arbitrary integer to a register.
    
    An implementation of an adding circuit described in de la Grand'rive and
    Hullo: "Knapsack Problem variants of QAOA for battery revenue
    optimisation" (2019).
    
    The circuit takes a quantum register of n qubits storing some binary
    number and adds some number x to that number. For this a carry register of
    size n-1 is required. It is assumed that x may be expressed in at most n
    (qu)bits. If this is not the case, weird things might happen. This class
    also contains a controlled version of the circuit, where the state of a
    single control qubit decides whether an addition takes place or not, as
    well as a reversed version of the circuit for uncomputing.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    control: controls the addition (1 qubit, optional)
    main: the register to which the circuit adds (n qubits)
    carry: auxillary register necessary for addition (n-1 qubits)
    
    Note:
    The control qubit is only included as part of the registers if
    controlled == True. Otherwise this circuit is only defined on main and
    carry registers.
    """
    
    def __init__(self, n, x, controlled = False, uncompute = False):
        """
        Initialize the circuit.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        x (int): the number being added
        
        Keyword Arguments:
        controlled (bool): whether or not you want the controlled version of
            the circuit (default: False, i.e. no control qubit)
        uncompute (bool): whether or not you want the reversed version of the
            circuit for uncomputing (default: False, i.e. regular version of
            the circuit)
        
        Note:
        The control qubit is only included as part of the registers if
        controlled == True. Otherwise this circuit is only defined on main and
        carry registers.
        """
        # Create registers
        main = QuantumRegister(n, name="main")
        carry = QuantumRegister(n-1, name="carry")
        if controlled:
            control = QuantumRegister(1, name="control")
            registers = [control, main, carry]
            qubits = [control, *main, *carry]
        else:
            registers = [main, carry]
            qubits = [*main, *carry]
        
        # Create QuantumCircuit
        name = f"Uncompute Add {x}" if uncompute else f"Add {x}"
        super().__init__(*registers, name=name)
        
        # Build circuit
        if uncompute:
            self.build_uncompute(n, x, controlled, qubits)
        else:
            self.build_regular(n, x, controlled, qubits)
            
    def build_regular(self, n, x, controlled, qubits):
        """
        Build the circuit in regular order.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        x (int): the number being added
        controlled (bool): whether or not you want the controlled version of
            the circuit (True means controlled version)
        qubits (list): list of all the qubits that the circuit acts on
        """
        binary = list(map(int, bin(x)[2:]))
        padded_binary = [0] * (n - len(binary)) + binary
        for k, x_k in enumerate(reversed(padded_binary)):
            if x_k:
                kadder = KAdder(n, k, controlled=controlled)
                super().append(kadder.to_instruction(), qubits)
                
    def build_uncompute(self, n, x, controlled, qubits):
        """
        Build the circuit in reverse order for uncomputing.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        x (int): the number being added
        controlled (bool): whether or not you want the controlled version of
            the circuit (True means controlled version)
        qubits (list): list of all the registers that the circuit acts on
        """
        binary = list(map(int, bin(x)[2:]))
        padded_binary = [0] * (n - len(binary)) + binary
        for k, x_k in reversed(list(enumerate(reversed(padded_binary)))):
            if x_k:
                kadder = KAdder(n, k, controlled=controlled, uncompute=True)
                super().append(kadder.to_instruction(), qubits)


class WeightCalculator(QuantumCircuit):
    """
    Circuit for calculating the weight of an item choice.
    
    An implementation of the weight calculation circuit described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019).
    
    The circuit takes a quantum register of N qubits storing some values
    representing a choice of items. Each item choice qubit controls an
    Adder circuit, adding the weight associated with that item to a weight
    register of n qubits. For this a carry register of size n-1 is required.
    n must be chosen large enough, s.t. the addition can take place without
    overflows. This class also contains a reversed version of the circuit for
    uncomputing.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    qchoices: the item choices (N qubits)
    qweight: stores weight of item choice (n qubits)
    qcarry: carry register for addition (n-1 qubits)
    """
    
    def __init__(self, n, weights, uncompute = False):
        """
        Initialize the circuit.
        
        Arguments:
        n (int): the size of the register for storing the weight
        weights (list): the weights of the items
        
        Keyword Arguments:
        uncompute (bool): whether or not you want the reversed version of the
            circuit for uncomputing (default: False, i.e. regular version of
            the circuit)
        """
        N = len(weights)

        qchoices = QuantumRegister(N, name="choices")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        registers = [qchoices, qweight, qcarry]

        name = "Uncompute Calculate Weight" if uncompute else "Calculate Weight" 
        super().__init__(*registers, name=name)
        
        if uncompute:
            self.build_uncompute(n, weights, *registers)
        else:
            self.build_regular(n, weights, *registers)  
        
    def build_regular(self, n, weights, qchoices, qweight, qcarry):
        """
        Build the circuit in regular order.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        weights (list): the weights of the items
        qchoices (QuantumRegister): register for item choices
        qweight (QuantumRegister): register for storing weight
        qcarry (QuantumRegister): carry register for addition
        """
        for qubit, weight in zip(qchoices, weights):
            adder = Adder(n, weight, controlled=True)
            super().append(adder.to_instruction(), [qubit, *qweight, *qcarry])
    
    def build_uncompute(self, n, weights, qchoices, qweight, qcarry):
        """
        Build the circuit in reversed order for uncomputing.
        
        Arguments:
        n (int): the size of the register to which this circuit should add
        weights (list): the weights of the items
        qchoices (QuantumRegister): register for item choices
        qweight (QuantumRegister): register for storing weight
        qcarry (QuantumRegister): carry register for addition
        """
        for qubit, weight in reversed(list(zip(qchoices, weights))):
            adder = Adder(n, weight, controlled=True, uncompute=True)
            super().append(adder.to_instruction(), [qubit, *qweight, *qcarry])


class LTChecker(QuantumCircuit):
    """
    Circuit for checking if the number in a register is less than 2^c.
    
    An implementation of a circuit described in de la Grand'rive and
    Hullo: "Knapsack Problem variants of QAOA for battery revenue
    optimisation" (2019).
    
    The circuit takes a quantum register of n qubits storing some binary
    number x and toggles a flag qubit if x is less than 2^c, where c
    is a non-negative integer. For this a carry register of
    size n-1 is required. It is assumed that c < n. If this is not the case,
    weird things might happen. This class also contains a reversed version of
    the circuit for uncomputing.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    qweight: the register which contains the number (n qubits)
    qcarry: auxillary register for size checking (n-1 qubits)
    qflag: flag qubit for storing the result (1 qubit)
    """

    def __init__(self, n, c, uncompute=False):
        """
        Initialize the circuit.
        
        Arguments:
        n (int): the size of the register for storing the number
        c (int): the exponent of the number with which the comparison should
            take place (compares with 2^c)
        
        Keyword Arguments:
        uncompute (bool): whether or not you want the reversed version of the
            circuit for uncomputing (default: False, i.e. regular version of
            the circuit)
        """
        qweight = QuantumRegister(n,  name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag = QuantumRegister(1, name="flag")
        registers = [qweight, qcarry, qflag]

        name = f"Uncompute Check < {2**c}" if uncompute else f"Check < {2**c}"
        super().__init__(*registers, name=name)
        
        if uncompute:
            self.build_uncompute(n, c, *registers)
        else:
            self.build_regular(n, c, *registers)
        
    def build_regular(self, n, c, qweight, qcarry, qflag):
        """
        Build the circuit in regular order.
        
        Arguments:
        n (int): the size of the register in which the number is stored
        c (int): the exponent of the number with which the comparison should
            take place (compares with 2^c)
        qweight (QuantumRegister): register for number
        qcarry (QuantumRegister): auxillary register for comparison
        qflag (QuantumRegister): flag register for result
        """
        super().x(qweight[c:])

        if c == n - 1:
            super().cx(qweight[c], qflag)
        elif c == n - 2:
            super().ccx(qweight[c], qweight[c+1], qflag)
        elif c <= n - 3:
            super().ccx(qweight[c], qweight[c+1], qcarry[0])

            for k in range(n - c - 3):
                super().ccx(qweight[c+2+k], qcarry[k], qcarry[k+1])

            super().ccx(qweight[n - 1], qcarry[n - c - 3], qflag)
            
            for k in reversed(range(n - c - 3)):
                super().ccx(qweight[c+2+k], qcarry[k], qcarry[k+1])
            
            super().ccx(qweight[c], qweight[c+1], qcarry[0])
        
        super().x(qweight[c:])
            
    def build_uncompute(self, n, c, qweight, qcarry, qflag):
        """
        Build the circuit in reversed order for uncomputing.
        
        Arguments:
        n (int): the size of the register in which the number is stored
        c (int): the exponent of the number with which the comparison should
            take place (compares with 2^c)
        qweight (QuantumRegister): register for number
        qcarry (QuantumRegister): auxillary register for comparison
        qflag (QuantumRegister): flag register for result
        """
        super().x(qweight[c:])
        
        if c == n - 1:
            super().cx(qweight[c], qflag)
        elif c == n - 2:
            super().ccx(qweight[c], qweight[c+1], qflag)
        elif c <= n - 3:
            super().ccx(qweight[c], qweight[c+1], qcarry[0])

            for k in range(n - c - 3):
                super().ccx(qweight[c+2+k], qcarry[k], qcarry[k+1])

            super().ccx(qweight[n - 1], qcarry[n - c - 3], qflag)
            
            for k in reversed(range(n - c - 3)):
                super().ccx(qweight[c+2+k], qcarry[k], qcarry[k+1])
            
            super().ccx(qweight[c], qweight[c+1], qcarry[0])
            
        super().x(qweight[c:])


##############################################################################
# Soft Constraint: Quadratic Penalty
##############################################################################


class QuadPhaseCirc(QuantumCircuit):
    """
    Phase seperation circuit for Knapsack QAOA with quadratic soft constraints.
    
    An implementation of the phase seperation circuit described in Roch et
    al.: "Cross Entropy Hyperparameter Optimization for Constrained Problem
    Hamiltonians Applied to QAOA" (2020). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qx: choice of items
    qy: one-hot encoding of weight
    
    Attributes:
    gamma (Parameter): phase seperation angle
    A (Parameter): prefactor of penalty term in the objective function
    B (Parameter): prefactor of the value term in the objective function
    """
    
    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
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
    
    An implementation of the mixer circuit described in Roch et al.: "Cross
    Entropy Hyperparameter Optimization for Constrained Problem Hamiltonians
    Applied to QAOA" (2020). The implementation is generalized to work for all
    instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    q: all qubits used by the circuit
    
    Attributes:
    beta (Parameter): mixing angle
    """
    
    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
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
    
    An implementation of the QAOA circuit described in Roch et al.: "Cross
    Entropy Hyperparameter Optimization for Constrained Problem Hamiltonians
    Applied to QAOA" (2020). The implementation is generalized to work for all
    instances of the KnapsackProblem class and arbitrary p.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qx: choice of items
    qy: one-hot encoding of weight
    
    Attributes:
    beta (Parameter): mixing angle
    gamma (Parameter): phase seperation angle
    A (Parameter): prefactor of penalty term in the objective function
    B (Parameter): prefactor of the value term in the objective function
    p (int): the number of times that phase seperation and mixing circuit are
        supposed to be applied
    """
    
    def __init__(self, problem: KnapsackProblem, p: int):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        """
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
    """
    QAOA for the Knapsack Problem with quadratic soft constraints.
    
    A class for grouping all necessary objects required for the implementation
    of QAOA for the 0-1 integer Knapsack Problem as described in Roch et al.:
    "Cross Entropy Hyperparameter Optimization for Constrained Problem
    Hamiltonians Applied to QAOA" (2020). The implementation is generalized to
    work for all suiting instances of the KnapsackProblem class and arbitrary
    p. 
    
    Attributes:
    problem (KnapsackProblem): the specific instance of the Knapsack Problem
    circuit (QuadQAOACirc): corresponding QAOA circuit
    
    Methods:
    objective_func: the specific objective function for this soft constraint
        approach.
    counts_to_choices: 
    """
    def __init__(self, problem: KnapsackProblem, p: int):
        """
        Create a QAOA circuit for the given problem.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        """
        self.problem = problem
        self.circuit = QuadQAOACirc(problem, p)
        
    def objective_func(self, bitstring: str, A: float, B: float):
        """
        Compute an objective function for the knapsack problem with quadratic soft constraints.
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


class PenaltyDephaser(QuantumCircuit):
    """
    Circuit for applying phase corresponding to weight penalty.
    
    An implementation of the penalty dephasing circuit described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019).
    
    The circuit takes a quantum register of n qubits storing the weight of the
    item choice (to which a constant offset has been added) and a flag qubit.
    If the flag qubit is set, a phase corresponding to the penalty described
    in above mentioned paper is applied.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    qweight: weight of the item choice (plus constant) (n qubits)
    qflag: flag qubit controlling the circuit (1 qubit)
    """
    
    def __init__(self, n, c):
        """
        Initialize the circuit.
        
        Arguments:
        n (int): the size of the register for storing the number
        c (int): the exponent of the number with which the weight comparison
            should has taken place (comparison with 2^c)
        """
        qweight = QuantumRegister(n, name="weight")
        qflag = QuantumRegister(1, name="flag")
        
        super().__init__(qweight, qflag, name="Dephase Penalty")

        self.alpha = Parameter("alpha")
        self.gamma = Parameter("gamma")
        
        for idx, qubit in enumerate(qweight):
            super().cp(2**idx * self.alpha * self.gamma, qflag, qubit)
    
        super().p(-2**c * self.alpha * self.gamma, qflag)


class ValueDephaser(QuantumCircuit):
    """
    Circuit for applying phase corresponding to value of item choice.
    
    An implementation of the value dephasing circuit described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019).
    
    The circuit takes a quantum register of N qubits storing the item choice.
    A phase corresponding to the value of the item choice is applied.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    qchoices: the item choices (N qubits)
    """
    
    def __init__(self, values):
        """
        Initialize the circuit.
        
        Arguments:
        values (list): the values of the items in the knapsack problem
        """
        N = len(values)
        qchoices = QuantumRegister(N)
        super().__init__(qchoices, name="Dephase Value")
        self.gamma = Parameter("gamma")
        for qubit, value in zip(qchoices, values):
            super().p(- self.gamma * value, qubit)


class LinPhaseCirc(QuantumCircuit):
    """
    Phase seperation circuit for Knapsack QAOA with linear soft constraints.
    
    An implementation of the phase seperation circuit described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qchoices: choice of items (N qubits)
    qweight: weight of item choice (n qubits)
    qcarry: auxillary register for e.g. addition (n-1 qubits)
    qflag: flag qubit signalling violation of constraints (1 qubit)
    
    Attributes:
    alpha (Parameter): prefactor of penalty term in the objective function
    gamma (Parameter): phase seperation angle
    """

    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1
        w0 = 2**c - problem.max_weight - 1

        qchoices = QuantumRegister(problem.N, name="choices")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag = QuantumRegister(1, name="flag")
        registers = [qchoices, qweight, qcarry, qflag]
        
        super().__init__(*registers, name="UPhase")

        self.alpha = Parameter("alpha")
        self.gamma = Parameter("gamma")

        super().x(qflag)

        valuecirc = ValueDephaser(problem.values)
        value_instruction = valuecirc.to_instruction({valuecirc.gamma: self.gamma})
        super().append(value_instruction, qchoices)

        super().append(WeightCalculator(n, problem.weights).to_instruction(), [*qchoices, *qweight, *qcarry])
        super().append(Adder(n, w0).to_instruction(), [*qweight, *qcarry])
        super().append(LTChecker(n, c).to_instruction(), [*qweight, *qcarry, qflag])

        penaltycirc = PenaltyDephaser(n, c)
        penalty_instruction = penaltycirc.to_instruction({penaltycirc.alpha: self.alpha, penaltycirc.gamma: self.gamma})
        super().append(penalty_instruction, [*qweight, qflag])

        super().append(LTChecker(n, c, uncompute=True).to_instruction(), [*qweight, *qcarry, qflag])
        super().append(Adder(n, w0, uncompute=True).to_instruction(), [*qweight, *qcarry])
        super().append(WeightCalculator(n, problem.weights, uncompute=True).to_instruction(), [*qchoices, *qweight, *qcarry])


class LinMixCirc(QuantumCircuit):
    """
    Mixing circuit for Knapsack QAOA with linear soft constraints.
    
    An implementation of the mixing circuit described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    q: choice of items (N qubits)
    
    Attributes:
    beta (Parameter): mixing angle
    """
    
    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
        # create registers
        q = QuantumRegister(problem.N)
        # create circuit
        super().__init__(q, name="UMix")
        
        # parameters for this circuit
        self.beta = Parameter("beta")
        
        # x-rotation of all qubits
        super().rx(2 * self.beta, q)


class LinQAOACirc(QuantumCircuit):
    """
    QAOA Circuit for Knapsack Problem with linear soft constraints.
    
    An implementation of the QAOA circuit described in de la Grand'rive and
    Hullo: "Knapsack Problem variants of QAOA for battery revenue
    optimisation" (2019). The implementation is generalized to work for all
    instances of the KnapsackProblem class and arbitrary p.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qchoices: choice of items (N qubits)
    qweight: weight of item choice (n qubits)
    qcarry: auxillary register for e.g. addition (n-1 qubits)
    qflag: flag qubit signalling violation of constraints (1 qubit)
    
    Attributes:
    alpha (Parameter): prefactor of penalty term in the objective function
    beta (Parameter): mixing angle
    gamma (Parameter): phase seperation angle
    p (int): the number of times that phase seperation and mixing circuit are
        supposed to be applied
    """
    
    def __init__(self, problem: KnapsackProblem, p: int):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        qchoices = QuantumRegister(problem.N, name="choices")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag = QuantumRegister(1, name="flag")
        registers = [qchoices, qweight, qcarry, qflag]
        qubits = [*qchoices, *qweight, *qcarry, qflag]

        super().__init__(*registers, name="LinQAOA")

        # Phase seperation circuit
        phase_circ = LinPhaseCirc(problem)
        # Mixing circuit
        mix_circ = LinMixCirc(problem)
        
        # Implementation of QAOA Circuit
        self.p = p

        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        
        self.alpha = Parameter("alpha")

        # initialize
        super().h(qchoices)

        # alternating application of phase seperation and mixing unitaries
        for gamma, beta in zip(self.gammas, self.betas):
            # application of phase seperation unitary
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.alpha: self.alpha,
            }
            phase_instruction = phase_circ.to_instruction(phase_params)
            super().append(phase_instruction, qubits)
            
            # application of mixing unitary
            mix_params = {
                mix_circ.beta: beta,
            }
            mix_instruction = mix_circ.to_instruction(mix_params)
            super().append(mix_instruction, qchoices)

        # measurement
        super().measure_all()


class LinQAOA():
    """
    QAOA for the Knapsack Problem with linear soft constraints.
    
    A class for grouping all necessary objects required for the implementation
    of QAOA for the 0-1 integer Knapsack Problem as described in de la
    Grand'rive and Hullo: "Knapsack Problem variants of QAOA for battery
    revenue optimisation" (2019). The implementation is generalized to work
    for all suiting instances of the KnapsackProblem class and arbitrary p. 
    
    Attributes:
    problem (KnapsackProblem): the specific instance of the Knapsack Problem
    circuit (QuadQAOACirc): corresponding QAOA circuit
    
    Methods:
    objective_func: the specific objective function for this soft constraint
        approach.
    """
    
    def __init__(self, problem: KnapsackProblem, p: int):
        """
        Create a QAOA circuit for the given problem.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        """
        self.problem = problem
        self.circuit = LinQAOACirc(problem, p)
        
    def objective_func(self, bitstring: str, alpha: float):
        """
        Compute an objective function for the knapsack problem with linear soft constraints.
        """
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        choices = np.array(bits[:self.problem.N])
        weight = choices.dot(self.problem.weights)
        penalty = - alpha * (weight - self.problem.max_weight) if weight > self.problem.max_weight else 0
        value = choices.dot(self.problem.values)
        return value + penalty


##############################################################################
# Hard Constraint: Random Walk
##############################################################################


class FeasibilityOracle(QuantumCircuit):
    """
    Circuit for checking feasibility of an item choice. 
    
    An implementation of the feasibility oracle described in Marsh and
    Wang: "A quantum walk-assisted approximate algorithm for bounded NP
    optimisation problems" (2019). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    The circuit takes an N qubit register representing an item choice,
    checks whether the constraints are satisfied, i.e. the item choice is
    considered feasible, and toggles a flag qubit accordingly.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit.
    
    Registers:
    qchoices: choice of items (N qubits)
    qweight: auxillary register for storing weights of item choices (n qubits)
    qcarry: auxillary register e.g. for addition (n-1 qubits)
    qflag: indicate whether choice in qx is valid (1 qubit)
    """
    
    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1
        w0 = 2**c - problem.max_weight - 1

        qchoices = QuantumRegister(problem.N, name="choices")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag = QuantumRegister(1, name="flag")
        registers = [qchoices, qweight, qcarry, qflag]
        
        super().__init__(*registers, name="FeasibilityOracle")
        
        super().append(WeightCalculator(n, problem.weights).to_instruction(), [*qchoices, *qweight, *qcarry])
        super().append(Adder(n, w0).to_instruction(), [*qweight, *qcarry])
        super().append(LTChecker(n, c).to_instruction(), [*qweight, *qcarry, qflag])
        super().append(Adder(n, w0, uncompute=True).to_instruction(), [*qweight, *qcarry])
        super().append(WeightCalculator(n, problem.weights, uncompute=True).to_instruction(), [*qchoices, *qweight, *qcarry])
        

class SingleQubitQuantumWalk(QuantumCircuit):
    """
    Circuit for single qubit quantum walk mixing. 
    
    An implementation of an improved version of the single qubit mixer
    described in Marsh and Wang: "A quantum walk-assisted approximate
    algorithm for bounded NP optimisation problems" (2019). The
    implementation is generalized to work for all instances of the
    KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qx: possible choices of items (N qubits)
    qweight: auxillary register for storing weights of item choices (n qubits)
    qcarry: auxillary register e.g. for addition (n-1 qubits)
    qflag_x: indicate whether choice in qx is valid (1 qubit)
    qflag_neighbor: indicate whether j-th neighbor of choice in qx is valid
        (1 qubit)
    qflag_both: indicate whether both choice in qx and its j-th neigbor are
        valid (1 qubit)
    
    Attributes:
    beta (Parameter): mixing angle
    """
    
    def __init__(self, problem: KnapsackProblem, j: int):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        j (int): number of the qubit which should be mixed
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        qx = QuantumRegister(problem.N, name="x")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag_x = QuantumRegister(1, name="v(x)")
        qflag_neighbor = QuantumRegister(1, name="v(n_j(x))")
        qflag_both = QuantumRegister(1, name="v_j(x)")
        
        registers = [qx, qweight, qcarry, qflag_x, qflag_neighbor, qflag_both]
        qubits = [*qx, *qweight, *qcarry, qflag_x, qflag_neighbor, qflag_both]
        
        super().__init__(*registers, name=f"SingleQubitQuantumWalk_{j=}")
        
        self.beta = Parameter("beta")
        
        super().append(FeasibilityOracle(problem).to_instruction(), [*qx, *qweight, *qcarry, qflag_x])
        super().x(qx[j])
        super().append(FeasibilityOracle(problem).to_instruction(), [*qx, *qweight, *qcarry, qflag_neighbor])
        super().x(qx[j])
        super().ccx(qflag_x, qflag_neighbor, qflag_both)
        
        super().crx(2 * self.beta, qflag_both, qx[j])
        
        super().ccx(qflag_x, qflag_neighbor, qflag_both)
        super().x(qx[j])
        super().append(FeasibilityOracle(problem).to_instruction(), [*qx, *qweight, *qcarry, qflag_neighbor])
        super().x(qx[j])
        super().append(FeasibilityOracle(problem).to_instruction(), [*qx, *qweight, *qcarry, qflag_x])
        
        
class QuantumWalkMixer(QuantumCircuit):
    """
    Mixing circuit for Knapsack QAOA with hard constraints.
    
    An implementation of the mixing circuit described in Marsh and
    Wang: "A quantum walk-assisted approximate algorithm for bounded NP
    optimisation problems" (2019). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qx: possible choices of items (N qubits)
    qweight: auxillary register for storing weights of item choices (n qubits)
    qcarry: auxillary register e.g. for addition (n-1 qubits)
    qflag_x: indicate whether choice in qx is valid (1 qubit)
    qflag_neighbor: indicate whether j-th neighbor of choice in qx is valid
        (1 qubit)
    qflag_both: indicate whether both choice in qx and its j-th neigbor are
        valid (1 qubit)
    
    Attributes:
    beta (Parameter): mixing angle
    """
    
    def __init__(self, problem: KnapsackProblem, m: int):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        m (int): degree of trotterization
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        qx = QuantumRegister(problem.N, name="x")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag_x = QuantumRegister(1, name="v(x)")
        qflag_neighbor = QuantumRegister(1, name="v(n_j(x))")
        qflag_both = QuantumRegister(1, name="v_j(x)")
        
        registers = [qx, qweight, qcarry, qflag_x, qflag_neighbor, qflag_both]
        qubits = [*qx, *qweight, *qcarry, qflag_x, qflag_neighbor, qflag_both]
        
        super().__init__(*registers, name=f"QuantumWalkMixer_{m=}")
        
        self.beta = Parameter("beta")
        
        for __ in range(m):
            for j in range(problem.N):
                jwalk = SingleQubitQuantumWalk(problem, j)
                super().append(jwalk.to_instruction({jwalk.beta: self.beta / m}), qubits)
                
                
class QuantumWalkPhaseCirc(QuantumCircuit):
    """
    Phase seperation circuit for Knapsack QAOA with hard constraints.
    
    An implementation of the phase seperation circuit described in Marsh and
    Wang: "A quantum walk-assisted approximate algorithm for bounded NP
    optimisation problems" (2019). The implementation is generalized
    to work for all instances of the KnapsackProblem class.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qchoices: choice of items (N qubits)
    
    Attributes:
    gamma (Parameter): phase seperation angle
    """
    
    def __init__(self, problem: KnapsackProblem):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        """
        qchoices = QuantumRegister(problem.N, name="choices")
        super().__init__(qchoices, name="UPhase")
        self.gamma = Parameter("gamma")
        
        valuecirc = ValueDephaser(problem.values)
        value_instruction = valuecirc.to_instruction({valuecirc.gamma: self.gamma})
        super().append(value_instruction, qchoices)
        
        
class QuantumWalkQAOACirc(QuantumCircuit):
    """
    QAOA Circuit for Knapsack Problem with hard constraints.
    
    An implementation of the QAOA circuit described in Marsh and Wang: "A
    quantum walk-assisted approximate algorithm for bounded NP optimisation
    problems" (2019). The implementation is generalized to work for all
    instances of the KnapsackProblem class and arbitrary p and m.
    
    Inherits from the QuantumCircuit class from qiskit and the interface
    remains mainly unchanged; only initializes the circuit and adds free
    circuit parameters as attributes. Those are of type
    qiskit.circuit.Parameter and noted as Parameter in the following.
    
    Registers:
    qx: possible choices of items (N qubits)
    qweight: auxillary register for storing weights of item choices (n qubits)
    qcarry: auxillary register e.g. for addition (n-1 qubits)
    qflag_x: indicate whether choice in qx is valid (1 qubit)
    qflag_neighbor: indicate whether j-th neighbor of choice in qx is valid
        (1 qubit)
    qflag_both: indicate whether both choice in qx and its j-th neigbor are
        valid (1 qubit)
    
    Attributes:
    beta (Parameter): mixing angle
    gamma (Parameter): phase seperation angle
    p (int): the number of times that phase seperation and mixing circuit are
        supposed to be applied
    """
    
    def __init__(self, problem: KnapsackProblem, p: int, m: int):
        """
        Initialize the circuit.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        m (int): the degree of trotterization for the mixing operator
        """
        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        qx = QuantumRegister(problem.N, name="x")
        qweight = QuantumRegister(n, name="weight")
        qcarry = QuantumRegister(n-1, name="carry")
        qflag_x = QuantumRegister(1, name="v(x)")
        qflag_neighbor = QuantumRegister(1, name="v(n_j(x))")
        qflag_both = QuantumRegister(1, name="v_j(x)")
        
        registers = [qx, qweight, qcarry, qflag_x, qflag_neighbor, qflag_both]
        qubits = [*qx, *qweight, *qcarry, qflag_x, qflag_neighbor, qflag_both]
        
        super().__init__(*registers, name=f"QuantumWalkMixer_{m=}")
        
        self.p = p
        
        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        
        # Phase seperation circuit
        phase_circ = QuantumWalkPhaseCirc(problem)
        # Mixing circuit
        mix_circ = QuantumWalkMixer(problem, m)
        
        # alternating application of phase seperation and mixing unitaries
        for gamma, beta in zip(self.gammas, self.betas):
            # application of phase seperation unitary
            phase_params = {phase_circ.gamma: gamma}
            phase_instruction = phase_circ.to_instruction(phase_params)
            super().append(phase_instruction, qx)
            
            # application of mixing unitary
            mix_params = {mix_circ.beta: beta}
            mix_instruction = mix_circ.to_instruction(mix_params)
            super().append(mix_instruction, qubits)

        # measurement
        super().measure_all()


class QuantumWalkQAOA:
    """
    QAOA for the Knapsack Problem with hard constraints.
    
    A class for grouping all necessary objects required for the implementation
    of QAOA for the 0-1 integer Knapsack Problem as described in Marsh and
    Wang: "A quantum walk-assisted approximate algorithm for bounded NP
    optimisation problems" (2019). The implementation is generalized to work
    for all suiting instances of the KnapsackProblem class and arbitrary p. 
    
    Attributes:
    problem (KnapsackProblem): the specific instance of the Knapsack Problem
    circuit (QuadQAOACirc): corresponding QAOA circuit
    
    Methods:
    objective_func: the specific objective function for this soft constraint
        approach.
    """
    
    def __init__(self, problem: KnapsackProblem, p: int, m:int):
        """
        Create a QAOA circuit for the given problem.
        
        The implementation is generalized, s.t. it will work for any instances
        of a 0-1 integer Knapsack Problem.
        
        Arguments:
        problem (KnapsackProblem): the instance of the knapsack problem that
            should be solved.
        p (int): the number of times that phase seperation and mixing circuit
            are supposed to be applied
        m (int): the degree of trotterization for the mixing operator
        """
        self.problem = problem
        self.circuit = QuantumWalkQAOACirc(problem, p, m)
        
    def objective_func(self, bitstring: str):
        """Computes an objective function for the knapsack problem with hard constraints."""
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        choices = np.array(bits[:self.problem.N])
        value = choices.dot(self.problem.values)
        return value


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

def simulate_circuit(parameter_values, transpiled_circuit: QuantumCircuit, variable_parameters: list, fixed_parameters: dict = None, backend = None, shots = 512, **kwargs):
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

def approximation_ratio(counts, best_known_solutions):
    """
    Calculates the approximation ratio of a solution.
    
    Here the approximation ratio refers to the percentage of sampled bitstrings
    that are one of the best known solutions. This is based on the definition
    given in Roch et al.: "Cross Entropy Hyperparameter Optimization for
    Constrained Problem Hamiltonians Applied to QAOA". This definition differs
    from the one given in Hadfield: "Quantum Algorithms for Scientific
    Computing and Approximate Optimization".
    """
    num_total_counts = sum(counts.values())
    best_counts = {k: v for k, v in counts.items() if k in best_known_solutions}
    num_best_counts = sum(best_counts.values())
    return num_best_counts / num_total_counts

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
    
    # Calculate and print the approximation ratio
    print(f"Approximation Ratio: {approximation_ratio(choices, ['10',]):.2f}")
    # Note that the approximation ratio is not taken with regard to counts, but
    # to choices, which is not correct. This is simply done to test its
    # working.
    
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
    example1()

if __name__ == "__main__":
    main()
