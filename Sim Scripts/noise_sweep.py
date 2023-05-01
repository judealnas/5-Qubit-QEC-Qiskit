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

# %% Construct circuit
def build_circuit(init_state:str) -> qiskit.QuantumCircuit:
    q_reg = FiveQubitCode.q_reg
    a_reg = FiveQubitCode.a_reg
    c_reg = FiveQubitCode.c_reg

    qcirc = QuantumCircuit(q_reg, a_reg, c_reg)
    qcirc.initialize(init_state+'0'*4,q_reg)
    qcirc.save_density_matrix([q_reg[4]],'init_data')
    qcirc.compose(FiveQubitCode.encoder(),inplace=True)
    qcirc.save_density_matrix(q_reg,'init_logical')
    qcirc.barrier()
    qcirc.delay(1,list(q_reg))
    qcirc.barrier()
    qcirc.compose(FiveQubitCode.stab_meas(),inplace=True)
    qcirc.compose(FiveQubitCode.err_correct(),inplace=True)
    qcirc.save_density_matrix(q_reg,'final_logical')
    qcirc.barrier()
    qcirc.compose(FiveQubitCode.decoder(),inplace=True)
    qcirc.save_density_matrix([q_reg[4]],'final_data')

    return qcirc

def run_sweep(noise_func, simulator, init_state: str,
               x_data: tuple, z_data: tuple, datafile: str):
    """Simulate over the list of parameters given

    :param noise_func: Two parameter function that returns a NoiseModel
    :type noise_func: Func(float,float)->NoiseModel
    :param simulator: The Qiskit simulator to use
    :type simulator: AerSimulator()
    :param init_state: String representing initial state of data qubit
    :type init_state: Union['0','1','-','+','l','r']
    :param x_data: A tuple of x data label and x data
    :type x_data: tuple(str, np.ndarray)
    :param z_data: A tuple of y data label and y data
    :type z_data: tuple(str, np.ndarray)
    :param fileprefix: Prefix of NPZ file containing data
    :type fileprefix: str
    """
    qcirc = build_circuit(init_state)
    T_qcirc = qiskit.transpile(qcirc,simulator)

    pX_label, pXs = x_data
    pZ_label, pZs = z_data

    fids_logical = np.empty([pXs.size,pZs.size])
    fids_logical.fill(np.nan)
    fids_data = fids_logical.copy()

    for (x,px) in enumerate(pXs):
        for (z,pz) in enumerate(pZs):
            print(f"{datafile} Sim {x*pXs.size + z + 1}/{pXs.size*pZs.size}")

            noise_model = noise_func(px,pz)
            res = simulator.run(T_qcirc,noise_model=noise_model,shots=shots).result()
            
            # Get initial and final logical matrix
            rho_logical_i = res.data()['init_logical']
            rho_logical_f = res.data()['final_logical']
            rho_data_i = res.data()['init_data']
            rho_data_f = res.data()['final_data']

            fids_logical[x,z] = state_fidelity(rho_logical_i,rho_logical_f)
            fids_data[x,z] = state_fidelity(rho_data_i,rho_data_f)

            # Calculate fidelity and save to file
            np.savez(datafile,shots=shots,
                     row_label=pX_label,row_data=pXs,
                     col_label=pZ_label,col_data=pZs,
                     fids_data=fids_data,fids_logical=fids_logical)

# %% Simulation parameters
# Probabilities to sweep
pXs = np.append(np.logspace(-3,0,20),np.linspace(0,1,20))
# pXs.put(0,0)
pXs.sort()
pZs = pXs.copy()

# Initial states
init_states = ['0','+','r']

# simulator = service.get_backend('ibmq_qasm_simulator')
simulator = Aer.get_backend('aer_simulator_density_matrix')
shots = 2**9

# %% Quantum error sweep
def quantum_error(px,pz):
    return get_noise(meas_error=[1,0], quantum_error=[px,pz])

x_data = ('px', pXs)
z_data = ('pz', pZs)

for s in init_states:
    datafile = f'quant_err_sweep_{s}'
    print(f'Quantum Error Sweep {s}')
    run_sweep(quantum_error, simulator, s, x_data, z_data, datafile)

# %% Measurment error sweep
def measurement_error(eff: float, dc: float):
    return get_noise(meas_error=[eff,dc], quantum_error=[0,0])

x_data = ('eff', pXs)
z_data = ('dc', pZs)

for s in init_states:
    datafile = f'meas_err_sweep_{s}'
    print(f'Measurement Error Sweep {s}')
    run_sweep(measurement_error, simulator, s, x_data, z_data, datafile)
# %% Efficiency sweep
def efficiency_error(eff: float,p: float):
    return get_noise(meas_error=[eff,0], quantum_error=[p]*2)

x_data = ('eff', pXs)
z_data = ('p', pZs)

for s in init_states:
    datafile = f'eff_err_sweep_{s}'
    print(f'Efficiency Error Sweep {s}')
    run_sweep(efficiency_error, simulator, s, x_data, z_data, datafile)
# %% Dark count error sweep
def darkcount_error(dc: float, p: float):
    return get_noise(meas_error=[dc,0], quantum_error=[p]*2)

x_data = ('dc', pXs)
z_data = ('p', pZs)

for s in init_states:
    datafile = f'dc_err_sweep_{s}'
    print(f'Dark count Error Sweep {s}')
    run_sweep(darkcount_error, simulator, s, x_data, z_data, datafile)
# %%
