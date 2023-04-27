from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error

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

def get_noise(meas_error: list|NoiseModel, quantum_error: list|NoiseModel) -> NoiseModel:
    """Construct a noise model containing measurement errors and noisy quantum delay gates on all qubits.

    :param meas_error: A list of form [Eff, DC] where Eff is the detection efficiency and DC is the dark count probability. 
                       Alternatively specify a pre-instantiated ReadoutError object
    :type meas_error: list | NoiseModel
    :param quantum_error: A list of form [pX, pZ] where pX and pZ are respectively the bit and phase flip channel probabilities.
                       Alternatively specify a pre-instantiated QuantumError object
    :type quantum_error: list | NoiseModel
    :return: Noise model where the specified errors are applied to all measurements and all qubits via the 'delay' gate.
    :rtype: NoiseModel
    """
    if type(meas_error) == list:
        meas_error = two_param_meas_error(*meas_error)
    if type(quantum_error) == list:
        quantum_error = two_param_pauli_error(*quantum_error)
    
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(meas_error)
    noise_model.add_all_qubit_quantum_error(quantum_error, ['i','id','delay'])

    return noise_model