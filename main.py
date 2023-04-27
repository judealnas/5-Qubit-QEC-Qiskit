# %%
import qiskit
from qiskit.providers.aer.noise import NoiseModel, ReadoutError
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import state_fidelity
from FiveQubitCode import FiveQubitCode

# %%
def two_param_meas_error(eff: float, dc: float, atol: float=1e-8) -> ReadoutError:
    """Construct Qiskit measurement error model using detection and dark count probabilities

    :param eff: Detection probability. Probability of measuring 1 if 1 is true measurement
    :type eff: float
    :param dc: Dark count probability. Probability of measuring 1 if 0 is true measurement
    :type dc: float
    :param atol: tolerance, defaults to 1e-8
    :type atol: float, optional
    :return: Measurement error model
    :rtype: ReadoutError
    """
    return ReadoutError([[1-dc,  dc],  #[P(0|0), P(1|0)]
                         [1-eff, eff]] #[P(0|1), P(1|1)]
                         ,atol) 


def two_param_pauli_error(pX: float, pZ: float) -> pauli_error:
    """Construct a quantum Pauli error channel 
    composed of a bit-flip (X) and phase-flip (Z) 
    channel with their respective probabilities

    :param pX: Bit flip probability
    :type pX: float
    :param pZ: Phase flip probability
    :type pZ: float
    :return: Pauli error model
    :rtype: pauli_error
    """
    x_error = pauli_error([('X',pX),('I',1-pX)])
    z_error = pauli_error([('Z',pZ),('I',1-pZ)])

    return x_error.compose(z_error)

# %% Ideal case
meas_error = two_param_meas_error(1,0)
quantum_error = two_param_pauli_error(0.5,0)

noise_model = NoiseModel()
noise_model.add_all_qubit_readout_error(meas_error)
noise_model.add_all_qubit_quantum_error(quantum_error, ["i",'id','delay'])

steps = 1

circ_meas = FiveQubitCode.stab_meas()
circ_enc = FiveQubitCode.encoder()
circ_correct = FiveQubitCode.err_correct()
circ_dec = FiveQubitCode.decoder()

circ_enc.barrier()
qcirc = circ_meas.compose(circ_enc,front=True)
qcirc.barrier()
qcirc.compose(circ_correct, inplace=True)
qcirc.barrier()
for _ in range(steps):
    qcirc.delay(1)
qcirc.barrier()
qcirc.compose(circ_dec, inplace=True)

log_creg = qiskit.ClassicalRegister(1,name='logical')
qcirc.add_register(log_creg)
qcirc.measure(qcirc.qregs[0][4],log_creg)

print(qcirc)
# qcirc.draw(output='mpl',filename='./test.png')

# run the circuit with th noise model and extract the counts
res = execute(qcirc, Aer.get_backend('qasm_simulator'),noise_model=noise_model).result()
counts = res.get_counts()

print(counts)
        