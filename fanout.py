# !/usr/bin/python3.8

import numpy as np
import qiskit
from qiskit import *
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.providers.fake_provider import *
# from qiskit_aer.pulse import *
from qiskit.providers import *
# from qiskit_experiments.framework import ParallelExperiment
# from qiskit_experiments.library import StateTomography
from qiskit_aer import AerSimulator
import qiskit_aer.noise as NoiseModel

#simulator= AerSimulator()


#3 address qubits & noise model
bus = QuantumRegister(1, name='bus')
ad = QuantumRegister(2, name='ad')
qram = QuantumRegister(21, name='qram')
# output = ClassicalRegister(1, name='output')
#data = ClassicalRegister(8, name='data')
qc = QuantumCircuit(bus,ad,qram)

##initialization
## x gate for 1
# qc.initialize(0,bus)
# # qc.initialize(1,ad[2])
# qc.initialize(1,qram[10])
# qc.initialize(0,ad[1])
# qc.initialize(1,ad[0])

qc.barrier()


#load address
# qc.cnot(ad[2], qram[10])
qc.cx(ad[1], qram[4])
qc.cx(ad[1], qram[16])
qc.cx(ad[0], qram[1])
qc.cx(ad[0], qram[7])
qc.cx(ad[0], qram[13])
qc.cx(ad[0], qram[19])

qc.barrier()


#retieve data
qc.h(bus)
qc.cswap(qram[10],qram[9],bus)
qc.x(qram[10])
qc.cswap(qram[10],qram[11],bus)
qc.x(qram[10])
qc.cswap(qram[4],qram[3],qram[9])
qc.x(qram[4])
qc.cswap(qram[4],qram[5],qram[9])
qc.x(qram[4])
qc.cswap(qram[16],qram[15],qram[11])
qc.x(qram[16]) 
qc.cswap(qram[16],qram[17],qram[11])
qc.x(qram[16])
qc.cswap(qram[1],qram[3],qram[0])
qc.x(qram[1])
qc.cswap(qram[1],qram[3],qram[2])
qc.x(qram[1])
qc.cswap(qram[7],qram[6],qram[5])
qc.x(qram[7])
qc.cswap(qram[7],qram[8],qram[5])
qc.x(qram[7])
qc.cswap(qram[13],qram[12],qram[15])
qc.x(qram[13])
qc.cswap(qram[13],qram[14],qram[15])
qc.x(qram[13])
qc.cswap(qram[19],qram[18],qram[17])
qc.x(qram[19])
qc.cswap(qram[19],qram[20],qram[17])
qc.x(qram[19])

qc.barrier()


#data


qc.barrier()


#retreive data
qc.cswap(qram[1],qram[3],qram[0])
qc.x(qram[1])
qc.cswap(qram[1],qram[3],qram[2])
qc.x(qram[1])
qc.cswap(qram[7],qram[6],qram[5])
qc.x(qram[7])
qc.cswap(qram[7],qram[8],qram[5])
qc.x(qram[7])
qc.cswap(qram[13],qram[12],qram[15])
qc.x(qram[13])
qc.cswap(qram[13],qram[14],qram[15])
qc.x(qram[13])
qc.cswap(qram[19],qram[18],qram[17])
qc.x(qram[19])
qc.cswap(qram[19],qram[20],qram[17])
qc.x(qram[19])
qc.cswap(qram[4],qram[3],qram[9])
qc.x(qram[4])
qc.cswap(qram[4],qram[5],qram[9])
qc.x(qram[4])
qc.cswap(qram[16],qram[15],qram[11])
qc.x(qram[16]) 
qc.cswap(qram[16],qram[17],qram[11])
qc.x(qram[16])
qc.cswap(qram[10],qram[9],bus)
qc.x(qram[10])
qc.cswap(qram[10],qram[11],bus)
qc.x(qram[10])
qc.h(bus)


# qc.measure(bus,output)
# qc.measure_all()

print(qc.draw())


# backend=BasicAer.get_backend('qasm_simulator')
# job = backend.run(transpile(qc,backend))
# job.result().get_counts(qc)


# # Example error probabilities
target_state = qi.Statevector.from_instruction(qc)
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
p_gate1 = 0.05

# # QuantumError objects

error_gate1 = pauli_error([('Z',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# # Add errors to noise model
noise_bit_flip = NoiseModel()
###all qubit error
# noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
# noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
###bus error
noise_bit_flip.add_quantum_error(error_gate1, ["u1", "u2", "u3"], bus)
print(noise_bit_flip)

F_bell = 0

sim_noise = AerSimulator(noise_model=noise_bit_flip)
qc.save_statevector()
for x in range(1):
    # Grab results from the job
    result = sim_noise.run(qc).result()
    rho_fit= result.get_statevector(qc)
    F_bell += qi.state_fidelity(rho_fit, target_state)
    print(qi.state_fidelity(rho_fit, target_state))
F_bell = F_bell / 2

print('State Fidelity: F = {:.6f}'.format(F_bell))

