# Using ideal measurements, sweep 
# the X and Z error probabilities,
# calculating the final state fidelities 
# given initial states of logical |0>,|+> and |+i>
# %%
import sys
import os
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.realpath('..'))
import qiskit
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.errors import pauli_error
from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.result import Result
from qiskit.quantum_info import state_fidelity, DensityMatrix
import numpy as np
from MyErrorModels import get_noise
from FiveQubitCode import FiveQubitCode

# %%
# Probabilities to sweep
pXs = np.append(np.float_power(10,np.arange(-3,0)),np.arange(0.125,1.125,0.125))
pXs.put(0,0)
pZs = pXs

# Initial states
init_states = ['0','+','r']

simulator = Aer.get_backend('aer_simulator_density_matrix')
shots = 2**9

# Construct circuit
def build_circuit(init_state:str) -> qiskit.QuantumCircuit:
    q_reg = FiveQubitCode.q_reg
    a_reg = FiveQubitCode.a_reg
    c_reg = FiveQubitCode.c_reg

    qcirc = QuantumCircuit(q_reg, a_reg, c_reg)
    qcirc.initialize(init_state+'0'*4,q_reg)
    qcirc.compose(FiveQubitCode.encoder(),inplace=True)
    qcirc.save_density_matrix(q_reg,'init_logical')
    qcirc.barrier()
    qcirc.delay(1,list(q_reg))
    qcirc.barrier()
    qcirc.compose(FiveQubitCode.stab_meas(),inplace=True)
    qcirc.compose(FiveQubitCode.err_correct(),inplace=True)
    qcirc.save_density_matrix(q_reg,'final_logical')

    return qcirc

# %%
for s in init_states:
    datafile = f'qerr_sweep_{s}'
    qcirc = build_circuit(s)
    T_qcirc = qiskit.transpile(qcirc,simulator)
    T_qcirc.draw(output='mpl')
    
    fids = np.empty([pXs.size,pZs.size])

    for (x,px) in enumerate(pXs):
        for (z,pz) in enumerate(pZs):
            print(f"{s} Sim {x*pXs.size + z + 1}/{fids.size}")

            noise_model = get_noise(meas_error=[1,0], quantum_error=[px,pz])
            res = simulator.run(T_qcirc,noise_model=noise_model,shots=shots).result()
            
            # Get initial and final logical matrix
            rho_logical_i = res.data()['init_logical']
            rho_logical_f = res.data()['final_logical']

            # Calculate fidelity and save to file
            fids[x,z] = state_fidelity(rho_logical_i,rho_logical_f)
            np.savez(datafile,shots=shots,row_data=pXs,col_data=pZs,fids=fids)
