'''
Script to simulate the effects of repeated measurements of syndromes in the 5-qubit
code in the presence of measurement errors.

The simulation is done in three parts:
1) Given a quantum error configuration, prepare an ensemble of noisy states by simulating
   the encoder, error channel, and a single measurement cycle for a statistically signficant # of shots.
   Save the final states from each shot.
   The single measurement cycle is to project the noisy logical state in preparation for repeated 
   measurements.
   Otherwise, repeating the measurements in Qiskit would implicitly reprepare the state in a
   superposition, which is a non-physical situation.

2) For each shot from 1), simulate a circuit that contains a single measurement cycle for 
   as many shots as desired repeated measurements. Save the measurements of each shot instead
   of Qiskit's average

3) Using the LUT and QuTiP, determine the average syndrome, apply the correction, and calculate
   fidelity

4) Repeat 1)-3) for each new parameter configuration

The principle behind this simulation assumes that error only occurs during the first simulation.
Following that, we are free to measure the state as many times as we desire without introducing 
more quantum noise. 
In such case, by central limit theorem, a sufficient number of measurements will produce a 
distribution of syndrome measurements centered about the true syndrome, assuming that measurement
error are sufficiently small.
'''
# %%
import sys
import os
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.realpath('..'))

import qiskit
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.errors import pauli_error
from qiskit import execute, Aer, transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Parameter
from qiskit.result import Result
from qiskit.quantum_info import state_fidelity, DensityMatrix, Operator, partial_trace

import numpy as np
from MyErrorModels import get_noise
from FiveQubitCode import FiveQubitCode

from qutip import sigmax, sigmay, sigmaz
from qutip.qip.operations import gate_expand_1toN

from matplotlib import pyplot as plt
import matplotlib as mpl

from collections.abc import Callable
# %%
def prep_state_circuit(init_state:str) -> qiskit.QuantumCircuit:
    q_reg = FiveQubitCode.q_reg
    a_reg = FiveQubitCode.a_reg
    c_reg = FiveQubitCode.c_reg

    qcirc = QuantumCircuit(q_reg,a_reg,c_reg)
    qcirc.initialize(init_state+'0'*4,q_reg)
    qcirc.save_density_matrix([q_reg[4]],'init_data')
    qcirc.compose(FiveQubitCode.encoder(),inplace=True)
    qcirc.save_density_matrix(q_reg,'init_logical')
    qcirc.barrier()
    qcirc.delay(1,list(q_reg))
    qcirc.compose(FiveQubitCode.stab_meas(), inplace=True)
    qcirc.save_density_matrix(q_reg,'noisy_logical',pershot=True)
    
    return qcirc

def meas_circuit(init_rho: DensityMatrix) -> QuantumCircuit:
    q_reg = FiveQubitCode.q_reg
    a_reg = FiveQubitCode.a_reg
    c_reg = FiveQubitCode.c_reg

    # Initial ancilla state is density matrix 
    # of single qubit in 0 state
    rho_0 = DensityMatrix([[1,0],[0,0]])
    rho = rho_0.tensor(init_rho)

    qcirc = QuantumCircuit(q_reg,a_reg,c_reg)
    qcirc.set_density_matrix(rho)
    qcirc.compose(FiveQubitCode.stab_meas(), inplace=True)

    return qcirc

def correct_rho(rho: DensityMatrix, synd: str|int) -> DensityMatrix:
    """Apply a correction to rho given a syndrome.

    :param rho: Density matrix to correct
    :type rho: DensityMatrix
    :param synd: A decimal number or binary string representing syndrome result
    :type synd: str | int
    :return: Corrected density matrix
    :rtype: DensityMatrix
    """
    # If binary string passed, convert to decimal
    if type(synd) == str:
        synd = int(synd,2)

    # No correction necessary if syndrome is 0
    if synd == 0:
        return rho
    
    # Lookup correction string from look up table
    synd_lut: dict = FiveQubitCode.synd_LUT()
    op_str: str = synd_lut[synd]

    pauli_dict = {'X': sigmax, 'Y': sigmay, 'Z': sigmaz}
    
    # Corrections are all weight one.
    # Find where and which non-identity character occurs.
    for s in ['X','Y','Z']:
        try:
            n = op_str.index(s)
        except ValueError:
            pass
        else:
            # Stop searching if successful
            break

    # Expand single qubit correction to the 5 qubit Hilbert space
    # Note that since the op strings in the syndrom LUT are 
    # in Qiskit's big-endian format and that QuTiP is in little-endian
    # format, no change in index is necessary.
    op = np.array(gate_expand_1toN(pauli_dict[s](), len(op_str), n))
    rho = np.array(rho)
    return DensityMatrix(op @ (rho @ op))

def sim_reps(num_reps: int, num_shots:int, init_state: str, noise_model: NoiseModel):
    """Simulate repeated syndrome measurements of the 5-qubit code. For :param num_shots: number
       of noisy logical states, measure each :param num_reps: times, use the most frequent result as
       the syndrome, decode, and calculate the fidelity averaged over the :param num_shots: different
       states.

    :param num_reps: Number of repeated syndrome measurements
    :type num_reps: int
    :param num_shots: Number of noisy states to average over
    :type num_shots: int
    :param init_state: Initial state
    :type init_state: str
    :param noise_model: Classical and quantum error noise model
    :type noise_model: NoiseModel
    :return: The average and std. dev fidelities between logical and data states in a dictionary of (mean, std dev) tuples
    :rtype: dict
    """

    # To obtain counts and states pershot, qasm_simulator must be used
    simulator = Aer.get_backend('qasm_simulator')

    # Prepare a num_shots number of noisy logical states
    T_qcirc1 = transpile(prep_state_circuit(init_state), simulator)
    sim1_opts = {
        'shots' : num_shots,
        'memory': True,
        'meas_level': 2, # don't know if I really need this
        'noise_model': noise_model
    }

    # Simulate
    res: Result = simulator.run(T_qcirc1,**sim1_opts).result()

    # Extract initial density matrix states
    rho_data_i = res.data()['init_data']
    rho_logical_i = res.data()['init_logical']
    rhos_logical_noisy = res.data()['noisy_logical']

    # Initialize histograms, i.e. list of dictionaries where keys
    # are binary strings and keys are the # of occurences
    pershot_hist = [{s:1} for s in res.get_memory()]

    # For each logical state, repeatedly measure, averaging the syndrome measurements. 
    # No memory is necessary as Qiskit automatically accumulates the count data over 
    # the shots
    for i,rho in enumerate(rhos_logical_noisy):
        T_qcirc2 = transpile(meas_circuit(rho),simulator)
        res2 = simulator.run(T_qcirc2,noise_model=noise_model,shots=num_reps).result()

        # iterate over the count frequencies,
        # incrementing values in the total histogram appropriately
        for (bin_str,freq) in res2.get_counts().items():
            pershot_hist[i][bin_str] = pershot_hist[i].get(bin_str,0) + freq

    # Find the most frequent syndrome
    # by first swapping the values and keys of the histograms.
    # The keys are now the frequencies. Find the max value and use as key
    # to obtain most frequent syndrome
    pershot_hist_swap = [{v:k for k,v in d.items()} for d in pershot_hist]
    synds = [d[ sorted(d.keys())[-1] ] for d in pershot_hist_swap]

    # Correct the rhos according to the most frequent syndrome
    rhos_logical_f = [correct_rho(rho,synd) for synd,rho in zip(synds,rhos_logical_noisy)]

    # Obtain final data state by decoding
    decode_op = Operator(FiveQubitCode.decoder())
    rhos_data_f = [partial_trace(rho.evolve(decode_op),range(4))]

    # Calculate logical and data state fidelities
    fids_logical = [state_fidelity(rho_data_i, rf) for rf in rhos_data_f]
    fids_data = [state_fidelity(rho_logical_i, rf) for rf in rhos_logical_f]

    # Calculate and return averages and std. devs
    return {
        'logical': (np.mean(fids_logical), np.std(fids_logical)),
        'data': (np.mean(fids_data), np.std(fids_data))
        }

def rep_sweep(noise_func: Callable[[float],NoiseModel], init_state: str,
              reps: list, params: list, param_label: str, num_shots: int, filename:str):
    
    # Initialize fidelity matrices
    mean_fids_logical = np.empty([reps.size,params.size])
    mean_fids_logical.fill(np.nan)
    std_fids_logical = mean_fids_logical.copy()
    mean_fids_data = mean_fids_logical.copy()
    std_fids_data = mean_fids_logical.copy()

    for (i,n) in enumerate(reps):
        for(j,param) in enumerate(params):
            print(f'{filename} Sim {i*reps.size + j + 1}/{mean_fids_logical.size}')

            noise_model = noise_func(param)
            dict_sim = sim_reps(num_reps=n, num_shots=num_shots,
                                init_state=init_state, noise_model=noise_model)
            
            mean_fids_logical[i,j], std_fids_logical[i,j] = dict_sim['logical']
            mean_fids_data[i,j], std_fids_data[i,j] = dict_sim['data']

            np.savez(filename,
                     row_label='reps', row_data=reps,
                     col_label=param_label, col_data=params,
                     mean_fids_logical=mean_fids_logical, std_fids_logical=std_fids_logical,
                     mean_fids_data=mean_fids_data, std_fids_data=std_fids_data)
# %% Sweep parameters
meas_params = np.append(np.logspace(-3,0,20),np.linspace(0,1,20))
ps = [0,1e-3,1e-2,0.1]
ns = np.arange(1,20)
shots = 2**9
# %% Efficiency sweep
for p in ps:
    eff_noise_model_func = lambda x: get_noise(meas_error=[x,0],quantum_error=[p]*2)
    rep_sweep(noise_func=eff_noise_model_func,
              init_state='0',
              reps=ns,
              params=meas_params,
              param_label='eff',
              num_shots=2**9,
              filename=f'eff_rep_sweep_p{p}')
# %% Dark count sweep
for p in ps:
    eff_noise_model_func = lambda x: get_noise(meas_error=[0,x],quantum_error=[p]*2)
    rep_sweep(noise_func=eff_noise_model_func,
              init_state='0',
              reps=ns,
              params=meas_params,
              param_label='dc',
              num_shots=2**9,
              filename=f'dc_rep_sweep_p{p}')

    
    