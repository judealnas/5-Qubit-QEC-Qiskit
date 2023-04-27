# %% Imports
import qiskit
import numpy as np
# %% Utility funcs
def rotate(l: list, n: int):
        return l[n:] + l[:n]

def str_transpose(l: list) -> list:
    """Transpose list of strings

    :param l: List of strings
    :type l: List[str]
    :return: Transposed list of strings
    :rtype: List[str]
    """

    return [''.join(t) for t in zip(*l)]

# %% Code class
class FiveQubitCode():
    stabilizers = ['XZZXI',
                   'IXZZX',
                   'XIXZZ',
                   'ZXIXZ']    
    q_reg = qiskit.QuantumRegister(len(stabilizers[0]), name='q')
    a_reg = qiskit.AncillaRegister(len(stabilizers), name='a')
    c_reg = qiskit.ClassicalRegister(len(stabilizers), name='synds')
    __synd_LUT = {}

    def encoder() -> qiskit.QuantumCircuit:
        """Construct encoder circuit for 5-qubit code. Qubit 0 is encoded qubit

        :return: _description_
        :rtype: qiskit.QuantumCircuit
        """
        num_qubits = len(FiveQubitCode.stabilizers[0])
        # q_reg = qiskit.QuantumRegister(num_qubits)
        q_reg = FiveQubitCode.q_reg
        qcirc = qiskit.QuantumCircuit(q_reg,name='Encoder')
        
        # Initializr data qubits to |+>
        qcirc.h(q_reg[:-1])
        
        # Add CNOTs
        for i in range(num_qubits-1):
            qcirc.cnot(q_reg[i],q_reg[-1])
        
        for i in range(num_qubits):
            qcirc.cz(q_reg[i],q_reg[(i+1)%5])
        
        return qcirc

    def decoder() -> qiskit.QuantumCircuit:
        """Construct 5-qubit code decoder, which is simply the encoder circuit reversed.

        :return: _description_
        :rtype: qiskit.QuantumCircuit
        """
        qcirc = FiveQubitCode.encoder()
        qcirc.name = 'Decoder'
        return qcirc.reverse_ops()

    def stab_meas() -> qiskit.QuantumCircuit:
        
        a_reg = FiveQubitCode.a_reg
        c_reg = FiveQubitCode.c_reg
        q_reg = FiveQubitCode.q_reg
        qcirc = qiskit.QuantumCircuit(q_reg,a_reg,c_reg,name='Measurement')

        # Prep ancilla in |+> state
        qcirc.h(a_reg)

        # Construct controlled unitary measurements
        for j,stab in enumerate(FiveQubitCode.stabilizers):

            if j != 0: qcirc.barrier()

            for i,s in enumerate(stab):
                if s != 'I':
                    if s == 'Z':
                        qcirc.cz(a_reg[j],q_reg[i])
                    elif s == 'X':
                        qcirc.cnot(a_reg[j],q_reg[i])
            
            # X-basis measurements
            qcirc.h(a_reg[j])
            qcirc.measure(a_reg[j],c_reg[j])  
        return qcirc    

    def err_correct() -> qiskit.QuantumCircuit:
        c_reg = FiveQubitCode.c_reg
        q_reg = FiveQubitCode.q_reg
        qcirc = qiskit.QuantumCircuit(q_reg,c_reg,name='Correction')

        LUT = FiveQubitCode.synd_LUT()
        for key,val in LUT.items():
            qcirc.pauli(val,q_reg).c_if(c_reg,key)

        return qcirc

    def synd_LUT() -> dict:
        if not FiveQubitCode.__synd_LUT:
            stabs_np = np.array([list(s) for s in FiveQubitCode.stabilizers])
            num_stabs, num_qubits = stabs_np.shape

            # Ignore the trivial no error syndrome by starting at 1
            synds = list(range(1,2**num_stabs))

            corrections = []
            for synd in synds:
                
                # Initial correction; to be changed
                correction = ['I']*num_qubits

                # Convert syndrome to (num_stabs)-bit binary
                bits = np.array([int(s) for s in format(synd,f'0{num_stabs}b')])

                anticomm_stabs = stabs_np[bits.astype(bool)]
                comm_stabs = stabs_np[np.logical_not(bits.astype(bool))]

                anti_sets = [set(a) for a in anticomm_stabs.T]
                comm_sets = [set(a) for a in comm_stabs.T]
                
                if sum(bits) == 1:
                    # If only one anti-commuting stabilizer, find which 
                    # operators only anti-commute on one qubit by 
                    # finding the difference between the anticommuting and 
                    # commuting sets and finding which index has a single, 
                    # non-identity element. Use the value of that element to determine the correctin
                    diff = [A-C for (A,C) in zip(anti_sets, comm_sets)]
                    try:
                        n = diff.index({'X'})
                        correction[n] = 'Z'
                    except ValueError:
                        n = diff.index({'Z'})
                        correction[n] = 'X'

                elif sum(bits) == 2:
                    # Find which qubit in the anti-commuting 
                    # set has only one unique operator.
                    n = [len(s) for s in anti_sets].index(1)
                    if 'X' in anti_sets[n]:
                        correction[n] = 'Z'
                    elif 'Z' in anti_sets[n]:
                        correction[n] = 'X'

                elif sum(bits) == 3:
                    # Find the qubit for which the identity 
                    # is the only commuting operator.
                    diff = [C-A for (A,C) in zip(anti_sets, comm_sets)]
                    n = diff.index({'I'})
                    correction[n] = 'Y'
                
                elif sum(bits) == 4:
                    # Find qubit with no identity ops in its stabilizers.
                    n = ['I' not in s for s in stabs_np.T].index(True)
                    correction[n] = 'Y'
                
                corrections.append(correction)

            # Form correction strings
            corrections = [''.join(c) for c in corrections]

            # Reverse bit strings and correction strings
            # to match Qiskit endianness
            synds = [int(format(n,f'0{num_stabs}b')[::-1],2) for n in synds]
            corrections = [c[::-1] for c in corrections]

            FiveQubitCode.__synd_LUT = dict(zip(synds,corrections))
        
        return FiveQubitCode.__synd_LUT

# %% 
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from qiskit.visualization import circuit_drawer
    

    circ_meas = FiveQubitCode.stab_meas()
    circ_enc = FiveQubitCode.encoder()
    circ_correct = FiveQubitCode.err_correct()
    circ_dec = FiveQubitCode.decoder()

    circ_enc.barrier()
    qcirc = circ_meas.compose(circ_enc,front=True)
    qcirc.barrier()
    qcirc.compose(circ_correct, inplace=True)
    qcirc.barrier()
    qcirc.compose(circ_dec, inplace=True)
    qcirc.draw(output='mpl',filename='./image.png')
    circ_correct.draw(output='mpl',filename='./circ_correct.png')
