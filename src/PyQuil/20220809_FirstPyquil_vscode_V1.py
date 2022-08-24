import numpy as np
from pyquil import get_qc, Program
from pyquil.gates import *
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare

prog = Program(
    Declare("ro", "BIT", 2),
    Z(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(10)

with local_forest_runtime():
    qvm = get_qc('9q-square-qvm')
    bitstrings = qvm.run(qvm.compile(prog)).readout_data.get("ro")


# construct a Bell State program
p = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(10)   


# run the program on a QVM
qc = get_qc('9q-square-qvm')
result = qc.run(qc.compile(p)).readout_data.get("ro")
print(result[0])
print(result[1])




p = Program()
p += X(0)

print(p)

p = Program()
ro = p.declare('ro', 'BIT', 1)
p += X(0)
p += MEASURE(0, ro[0])

qc = get_qc('1q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
executable = qc.compile(p)
result = qc.run(executable)
bitstrings = result.readout_data.get('ro')
print(bitstrings)



p = Program()
ro = p.declare('ro', 'BIT', 16)
theta = p.declare('theta', 'REAL')
print(p)


p = Program()
ro = p.declare('ro', 'BIT', 2)
p += H(0)
p += CNOT(0, 1)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])

print(p)

qubits = [5, 6, 7]
# ...
ro = p.declare('ro', 'BIT', len(qubits))
for i, q in enumerate(qubits):
    p += MEASURE(q, ro[i])

p.wrap_in_numshots_loop(10)

print(p)



qubit = 0

p = Program()
ro = p.declare("ro", "BIT", 1)
theta_ref = p.declare("theta", "REAL")

p += RX(np.pi / 2, qubit)
p += RZ(theta_ref, qubit)
p += RX(-np.pi / 2, qubit)

p += MEASURE(qubit, ro[0])
# Get a Quantum Virtual Machine to simulate execution
qc = get_qc("1q-qvm")
executable = qc.compile(p)

# Somewhere to store each list of results
parametric_measurements = []

for theta in np.linspace(0, 2 * np.pi, 200):
    # Set the desired parameter value in executable memory
    executable.write_memory(region_name='theta', value=theta)

    # Get the results of the run with the value we want to execute with
    bitstrings = qc.run(executable).readout_data.get("ro")

    # Store our results
    parametric_measurements.append(bitstrings)




p.wrap_in_numshots_loop(10)

with local_forest_runtime():
    qvm = get_qc('9q-square-qvm')
    bitstrings = qvm.run(qvm.compile(prog)).readout_data.get("ro")

bitstrings    



# Wavefunction simulator
from pyquil.api import WavefunctionSimulator
wf_sim = WavefunctionSimulator()
coin_flip = Program(H(0))
wf_sim.wavefunction(coin_flip)

coin_flip = Program(H(0))
wavefunction = wf_sim.wavefunction(coin_flip)
print(wavefunction)

from pyquil import Program
from pyquil.api import WavefunctionSimulator
from pyquil.quilatom import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
import numpy as np

# Define the new gate from a matrix
theta = Parameter('theta')
crx = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
    [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]
])

gate_definition = DefGate('CRX', crx, [theta])
CRX = gate_definition.get_constructor()

# Create our program and use the new parametric gate
p = Program()
p += gate_definition
p += H(0)
p += CRX(np.pi/2)(0, 1)



print(WavefunctionSimulator().wavefunction(p))




import numpy as np
from pyquil.quilbase import DefGate

ccnot_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
])

ccnot_conjugate_transpose = ccnot_matrix.transpose()
ccnot_inverse = np.linalg.inv(ccnot_matrix) 

unitary_check1 = np.matmul(ccnot_matrix,ccnot_conjugate_transpose)
unitary_check2 = np.matmul(ccnot_conjugate_transpose,ccnot_matrix)
unitary_check3 = np.matmul(ccnot_matrix,ccnot_inverse)

np.array_equal(unitary_check1, unitary_check2)
np.array_equal(unitary_check1, unitary_check3)
np.array_equal(unitary_check2, unitary_check3)

ccnot_size = ccnot_conjugate_transpose.shape[1]
identity = np.eye(8)


np.array_equal(unitary_check1, identity)
np.array_equal(unitary_check2, identity)
np.array_equal(unitary_check2, identity)



ccnot_gate = DefGate("CCNOT", ccnot_matrix)




from pyquil.quilbase import DefPermutationGate

ccnot_gate = DefPermutationGate("CCNOT", [0, 1, 2, 3, 4, 5, 7, 6])



from pyquil import Program
from pyquil.gates import *

p = Program(X(3))


from pyquil.quil import Pragma


p = Program(Pragma('INITIAL_REWIRING', ['"GREEDY"']))
p += X(3)



# Preferred method
p = Program()
p += X(0)
p += Y(1)
print(p)

# Multiple instructions in declaration
print(Program(X(0), Y(1)))

# A composition of two programs
print(Program(X(0)) + Program(Y(1)))

# Raw Quil with newlines
print(Program("X 0\nY 1"))

# Raw Quil comma separated
print(Program("X 0", "Y 1"))

# Chained inst; less preferred
print(Program().inst(X(0)).inst(Y(1)))




p = Program(X(0), Y(1), Z(2))
print(p)

print("We can fix by popping:")
p.pop()
print(p)





p = Program()
ro = p.declare('ro', 'BIT', 2)
p += X(0)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])
p.wrap_in_numshots_loop(5)

executable = qc.compile(p)
result = qc.run(executable)  # .run takes in a compiled program
bitstrings = result.readout_data.get("ro")
print(bitstrings)

import networkx as nx

from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.noise import decoherence_noise_with_asymmetric_ro

qubits = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]  # qubits are numbered by octagon
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),  # first octagon
         (1, 16), (2, 15),  # connections across the square
         (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (16, 17), (10, 17)] # second octagon

# Build the NX graph
topo = nx.from_edgelist(edges)
# You would uncomment the next line if you have disconnected qubits
# topo.add_nodes_from(qubits)
quantum_processor = NxQuantumProcessor(topo)
quantum_processor.noise_model = decoherence_noise_with_asymmetric_ro(quantum_processor.to_compiler_isa())  # Optional

from pyquil import Program
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator
wf_sim = WavefunctionSimulator()
coin_flip = Program(H(0))
wf_sim.wavefunction(coin_flip)

coin_flip = Program(H(0))
wavefunction = wf_sim.wavefunction(coin_flip)
print(wavefunction)


assert wavefunction[0] == 1 / np.sqrt(2)
# The amplitudes are stored as a numpy array on the Wavefunction object
print(wavefunction.amplitudes)
prob_dict = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes as a dict
print(prob_dict)
prob_dict.keys() # these store the bitstring outcomes
assert len(wavefunction) == 1 # gives the number of qubits






from pyquil.quil import Pragma, Program
from pyquil.api import get_qc
from pyquil.gates import CNOT, H

qc = get_qc("9q-square-qvm")

ep = qc.compile(Program(H(0), CNOT(0,1), CNOT(1,2)))

print(ep)



from pyquil.quil import Pragma, Program
from pyquil.api import get_qc
from pyquil.gates import CNOT, H

qc = get_qc("9q-square-qvm")

p = Program(H(0), CNOT(0,1), CNOT(1,2))

np = qc.compiler.quil_to_native_quil(p, protoquil=True)
print(np.metadata)

ep = qc.compiler.native_quil_to_executable(np)
print(ep)


qc = get_qc("9q-square-qvm", compiler_timeout=100) # 100 seconds



from pyquil import get_qc, Program


# If you have a reserved QPU, use it here
#qc = get_qc("Aspen-X")
# Otherwise use a QVM
qc = get_qc("8q-qvm")

# Likely you will have a more complex program:
p = Program("RX(pi) 0")

native_p = qc.compiler.quil_to_native_quil(p)

# The program will now have only native gates
print(native_p)
# And also a dictionary, with the above keys
print(native_p.native_quil_metadata["qpu_runtime_estimation"])


import networkx as nx
from pyquil import Program, get_qc
from pyquil.api._quantum_computer import _get_qvm_with_topology
from pyquil.gates import CZ

graph = nx.from_edgelist([(0, 1), (1, 2)])
qc = _get_qvm_with_topology(name="line", topology=graph)

p = Program(CZ(0, 2))
print(qc.compile(p))

