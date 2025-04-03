import strawberryfields as sf
import tensorflow as tf
import numpy as np
from strawberryfields.ops import MZgate, Ket

def FockBasis(state_list: list, cutoff_dim: int, modes: int, dtype: type = np.complex64, multiplex: bool = False): 
    """
    Create numpy.ndarray out of given indizes representing a specific state in Fock space
    
    Parameters
    ----------
    state_list : list
        list of tuple containing a states in the truncated Fock Space (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
    cutoff_dim : int  
        truncation of the Fock Space
    modes : int       
        number of modes in the truncated Fock Space
    dtype : type, optional     
        data type of the returned state (e.g. np.complex64 for standard states, bool for masks). The default is np.complex64
    multiplex: bool, optional
        Defines if multiple post selection states should be considered. The default is False.
       
    Returns
    -------
    ArrayLike
        complete numpy array to be handeld by Strawberryfields
        
    """
    state = []
    if multiplex:
        for _ in range(len(state_list[0])):
            state.append(np.zeros([cutoff_dim]*modes, dtype=dtype))
        
        for p in range(len(state_list)):
            for i in range(len(state_list[p])):
                state[i][state_list[p][i]] = 1
    else:
        for _ in range(len(state_list)):
            state.append(np.zeros([cutoff_dim]*modes, dtype=dtype))
        
        for i in range(len(state_list)):
            state[i][state_list[i]] = 1
    
    return np.array(state)

# Define the number of modes
NUM_MODES = 3

# Define initial state
#
# This definition, as well as the definition of FockBasis, allow for batch processing of multiple states. (e.g. [(1,0,0), (0,1,0), (0,0,1)])
# This is, however, not used in this example.
#
INITIAL_STATE = [(1, 0, 0)]

# Calculate the number of photons
num_photons = max([sum(INITIAL_STATE[i]) for i in range(len(INITIAL_STATE))])

# Define the initial state in the Fock basis
init_state = FockBasis(INITIAL_STATE, num_photons+1, NUM_MODES)[0]

# Calculate the number of free parameters
num_parameters = NUM_MODES * (NUM_MODES - 1)

# Define the Strawberry Fields program
program = sf.Program(NUM_MODES)
engine = sf.Engine("tf", backend_options={"cutoff_dim": num_photons + 1, "batch_size": None})

# Define the Strawberry Fields parameters
sf_params = []
para_names = ['theta', 'phi'] # Define the parameter names - theta for internal phaseshifter and phi for external phaseshifter
for i in range(num_parameters//len(para_names)):
    sf_params_names = ['{}_{}'.format(name, i) for name in para_names]
    sf_params.append(program.params(*sf_params_names))

sf_params = np.array(sf_params)

# Define Clements Universal Interferometer
structure = []
for _ in range(NUM_MODES//2):
    for i in range(2):
        for j in range(NUM_MODES-1):
            if j%2==i:
                structure.append((j, j+1))

if (NUM_MODES%2) != 0:
    for j in range(NUM_MODES-1):
        if j%2==0:
            structure.append((j, j+1))

# Define the Strawberry Fields Quantum Circuit
with program.context as q:
    
    # Initialize the state
    Ket(init_state) | q

    # Apply the Clements Universal Interferometer
    for p in range(num_parameters//len(para_names)):
        MZgate(sf_params[p, 0], sf_params[p, 1]) | tuple(q[i] for i in structure[p])

# Initialize the parameters
STDDEV = 0.01
MEAN = 0.0

phi = tf.random.normal(shape=[num_parameters//len(para_names)], stddev=STDDEV, mean=MEAN)
theta = tf.random.normal(shape=[num_parameters//len(para_names)], stddev=STDDEV, mean=MEAN)

weights = tf.convert_to_tensor([phi, theta])
weights = tf.Variable(tf.transpose(weights))

# Map weights to Strawberry Fields parameters
mapping = {}
for i in range(num_parameters//len(para_names)):
    mapping[sf_params[i,0].name] = weights[i,0]
    mapping[sf_params[i,1].name] = weights[i,1]

# Run the Strawberry Fields program
state = engine.run(program, args=mapping).state
prob = state.all_fock_probs()

# Print the evaluated probabilities
print(prob)

# Rest the Strawberry Fields eninge
engine.reset()
