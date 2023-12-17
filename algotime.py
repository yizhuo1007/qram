# !/usr/bin/python3.8

import numpy as np
import qiskit
import time
from qiskit import *
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.providers.fake_provider import *
# from qiskit_aer.pulse import *
from qiskit.providers import *
from qiskit_aer import AerSimulator
import qiskit_aer.noise as NoiseModel
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import DensityMatrix, Operator, Statevector


#######################################
##########2 address qram & 11 qubits
def oqram(qc: QuantumCircuit, ad0:QuantumRegister, ad1:QuantumRegister, bus:QuantumRegister, qram: QuantumRegister):
    # tin = time.time()
    # bus = QuantumRegister(1, name='bus')
    # #ad[1]==qram[4]
    # qram = QuantumRegister(8, name='qram')
    # # ad = QuantumRegister(1, name='ad')

    # qc = QuantumCircuit(ad0, ad1, bus,qram)
    ###initialization
    ### x gate for 1
    # qc.initialize(0,bus)
    # qram[4] = ad1
    # qc.barrier()
    ###load address
    qc.cx(ad0, qram[1])
    qc.cx(ad0, qram[6])
    qc.barrier()
    ###retieve data
    qc.h(bus)
    qc.cswap(ad1, qram[3], bus)
    qc.x(ad1)
    qc.cswap(ad1, qram[4], bus)
    qc.x(ad1)
    qc.cswap(qram[1], qram[0], qram[3])
    qc.x(qram[1])
    qc.cswap(qram[1], qram[2], qram[3])
    qc.x(qram[1])
    qc.cswap(qram[6], qram[5], qram[4])
    qc.x(qram[6])
    qc.cswap(qram[6], qram[7], qram[4])
    qc.x(qram[6])
    qc.barrier()
    ###data
    ##qram[0] ==x3
    ##qram[2] ==x2
    ##qram[6] ==x1
    ##qram[8] ==x0
    qc.barrier()
    ###retieve data
    qc.cswap(qram[6], qram[5], qram[4])
    qc.x(qram[6])
    qc.cswap(qram[6], qram[7], qram[4])
    qc.x(qram[6])
    qc.cswap(qram[1], qram[0], qram[3])
    qc.x(qram[1])
    qc.cswap(qram[1], qram[2], qram[3])
    qc.x(qram[1])
    qc.cswap(ad1, qram[3], bus)
    qc.x(ad1)
    qc.cswap(ad1, qram[4], bus)
    qc.x(ad1)
    qc.h(bus)
    qc.barrier()   

    # ###########啦啦啦啦啦啦啦啦啦
    k = np.random.randint(0,10,size=1)
    # print(k)
    m = 0
    for m in range (k[0]):
        qc.x(bus)
        qc.y(bus)
        qc.z(bus) 
    
    # tout = time.time()
    # print("t_in = ",tin)
    # print("t_out = ",tout)
    # print("t_query = ",tout-tin)
    # print(qc.draw())
    # target_state = qi.Statevector.from_instruction(qc)
    return 



# #############################
# ####Grover's algo
# grover = QuantumRegister(2, name='grover')
# bus = QuantumRegister(1, name='bus')
# qram = QuantumRegister(8, name='qram')
# circuit = QuantumCircuit(grover, bus, qram)

# t0 = time.time()
# circuit.h(grover[0])
# circuit.h(grover[1])

# ##oracle
# t1 = time.time()
# oqram(circuit, grover[0], grover[1], bus, qram)
# t2 = time.time()

# ##diffussion
# ## D = 2 * DensityMatrix.from_label('00') - Operator.from_label('II')
# ## grover_op = GroverOperator(oracle= oqram, zero_reflection= D)
# circuit.h(grover[0])
# circuit.h(grover[1])
# circuit.z(grover[0])
# circuit.z(grover[1])
# circuit.cz(grover[0],grover[1])
# circuit.h(grover[0])
# circuit.h(grover[1])
# circuit.h(grover[0])
# circuit.h(grover[1])
# t3 = time.time()

# print(circuit.draw())
# print("t_prepare = ",t1-t0)
# print("t_qram = ",t2-t1)
# print("t_compute = ",t3-t2)




# ###########################
# ####Simon's algo
# #####try 3 address qubits
# simon = QuantumRegister(1, name='simon')
# bus = QuantumRegister(1, name='bus')
# qram = QuantumRegister(8, name='qram')
# out = QuantumRegister(1, name='out')
# circuit = QuantumCircuit(simon, bus, qram, out)

# t0 = time.time()
# circuit.h(simon)

# t1 = time.time()
# oqram(circuit, simon, out, bus, qram)
# t2 = time.time()

# circuit.h(simon)
# t3 = time.time()

# print(circuit.draw())
# print("t_prepare = ",t1-t0)
# print("t_qram = ",t2-t1)
# print("t_compute = ",t3-t2)



# ##########################
# ####Bernstein-Vazirani algo
# bv = QuantumRegister(2, name='bv')
# bus = QuantumRegister(1, name='bus')
# qram = QuantumRegister(8, name='qram')
# out = QuantumRegister(1, name='out')
# # anci = QuantumRegister(1, name='anci')
# circuit = QuantumCircuit(bv, bus, qram, out)

# t0 = time.time()
# circuit.h(bv[0])
# circuit.h(bv[1])

# t1 = time.time()
# oqram(circuit, bv[0], bv[1], bus, qram)
# t2 = time.time()

# if bv[0] == '0':
#     circuit.i(out)
# else:
#     circuit.cx(bv[0], out)
# if bv[1] == '0':
#     circuit.i(out)
# else:
#     circuit.cx(bv[0], out)
# t3 = time.time()

# print(circuit.draw())
# print("t_prepare = ",t1-t0)
# print("t_qram = ",t2-t1)
# print("t_compute = ",t3-t2)



###########################
#####Hidden shift problem
#####3 address qubit
# gthyf67dcfxx

# # circuit.measure_all()

# # print(circuit)
# print("t_prepare = ",t1-t0)
# print("t_qram1 = ",t2-t1)
# print("t_compute1 = ",t3-t2)
# print("t_qram2 = ",t4-t3)
# print("t_compute2 = ",t5-t4)

# ##### 1000 turns
t_prepare = 0
t_qram1 = 0
t_compute1 = 0
t_qram2 = 0
t_compute2 = 0
for x in range(1000):
    hsp = QuantumRegister(2, name='hsp')
    bus = QuantumRegister(1, name='bus')
    qram = QuantumRegister(8, name='qram')
    circuit = QuantumCircuit(hsp, bus, qram)
    t0 = time.time()
    circuit.h(hsp[1])
    k = np.random.randint(15,25,size=1)
    m = 0
    for m in range (k[0]):
        circuit.x(bus)
        circuit.y(bus)
        circuit.z(bus)
    t1 = time.time()
    oqram(circuit, hsp[0], hsp[1], bus, qram)
    t2 = time.time()
    circuit.z(hsp[0])
    k = np.random.randint(15,25,size=1)
    m = 0
    for m in range (k[0]):
        circuit.x(bus)
        circuit.y(bus)
        circuit.z(bus)
    t3 = time.time()
    oqram(circuit, hsp[0], hsp[1], bus, qram)
    t4 = time.time()
    circuit.h(hsp[1])
    k = np.random.randint(15,25,size=1)
    m = 0
    for m in range (k[0]):
        circuit.x(bus)
        circuit.y(bus)
        circuit.z(bus)
    t5 = time.time()
    t_prepare +=t1-t0
    t_qram1 += t2-t1
    t_compute1 += t3-t2
    t_qram2 += t4-t3
    t_compute2 += t5-t4
t_prepare = t_prepare / 1001
t_qram1 = t_qram1 / 1001
t_compute1 = t_compute1 / 1001
t_qram2 = t_qram2 / 1001
t_compute2 = t_compute2 / 1001
print("t_prepare = ",t_prepare)
print("t_qram1 = ",t_qram1)
print("t_compute1 = ",t_compute1)
print("t_qram2 = ",t_qram2)
print("t_compute2 = ",t_compute2)


# #########################
# ####all qubit flip noise model
# target_state = qi.Statevector.from_instruction(circuit)
# # # Example error probabilities
# from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
#     pauli_error, depolarizing_error, thermal_relaxation_error)
# p_gate1 = 0.05

# # # QuantumError objects

# error_gate1 = pauli_error([('Z',p_gate1), ('I', 1 - p_gate1)])
# error_gate2 = error_gate1.tensor(error_gate1)

# # # Add errors to noise model
# noise_bit_flip = NoiseModel()
# noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
# noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

# print(noise_bit_flip)

# F_bell = 0

# sim_noise = AerSimulator(noise_model=noise_bit_flip)
# circuit.save_statevector()
# for x in range(100):
#     # Grab results from the job
#     result = sim_noise.run(circuit).result()
#     rho_fit= result.get_statevector(circuit)
#     F_bell += qi.state_fidelity(rho_fit, target_state)
#     # print(qi.state_fidelity(rho_fit, target_state))
# F_bell = F_bell / 101

# print('State Fidelity: F = {:.6f}'.format(F_bell))



###########################
######simulator 不好用 Q A Q
## Initialize on local simulator
# circuit.measure_all()
# shots = 1000
# sim_backend = BasicAer.get_backend("qasm_simulator")
# job = execute(circuit, sim_backend, shots=shots)
# result = job.result()

# counts = result.get_counts(circuit)

# qubit_strings = [format(i, "04b") for i in range(2**4)]
# print("Probabilities from simulator: ")
# print([format(counts.get(s, 0) / shots, ".3f") for s in qubit_strings])



#########################
######fake backend
backend = FakeGeneva()
# backend = FakeAuckland()
# backend = FakeGeneva()
# backend = FakeGuadalupeV2()
# backend = FakeHanoiV2()
# backend = FakeKolkataV2()
# backend = FakeMelbourneV2()
# backend = FakeMontrealV2()
# backend = FakeMumbaiV2()
# backend = FakePrague()
# backend = FakeSherbrooke()
# backend = FakeSydneyV2()
# backend = FakeTorontoV2()
# backend = FakeWashingtonV2()
# Transpile the ideal circuit to a circuit that can be directly executed by the backend
transpiled_circuit = transpile(circuit, backend)
print(backend)
# transpiled_circuit.draw('mpl')

# # Run the transpiled circuit using the simulated fake backend
# job = backend.run(transpiled_circuit)
# job.wait_for_final_state()
# counts = job.result(0).get_counts()
# # plot_histogram(counts)
# # print(counts)
