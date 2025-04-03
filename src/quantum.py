import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

class ClementCircuit:
    @staticmethod
    def fock_basis(state_list, cutoff_dim, modes, dtype=np.complex64, multiplex=False):
        """
        Create numpy.ndarray out of given indices representing a specific state in Fock space.
        
        Parameters
        ----------
        state_list : list
            List of tuples representing states in the truncated Fock space.
        cutoff_dim : int  
            Truncation of the Fock space.
        modes : int       
            Number of modes in the truncated Fock space.
        dtype : type, optional     
            Data type of the returned state.
        multiplex: bool, optional
            Whether to consider multiple post-selection states.
       
        Returns
        -------
        numpy.ndarray
            Complete numpy array to be handled by Strawberry Fields.
        """
        state = []
        if multiplex:
            for _ in range(len(state_list[0])):
                state.append(np.zeros([cutoff_dim] * modes, dtype=dtype))
            for p in range(len(state_list)):
                for i in range(len(state_list[p])):
                    state[i][state_list[p][i]] = 1
        else:
            for _ in range(len(state_list)):
                state.append(np.zeros([cutoff_dim] * modes, dtype=dtype))
            for i in range(len(state_list)):
                state[i][state_list[i]] = 1
        return np.array(state)

    def __init__(self, num_modes=3, initial_state=[(1, 0, 0)], stddev=0.01, mean=0.0, config=None):
        """
        Initialize the MemristorCircuit.
        
        Parameters
        ----------
        num_modes : int, optional
            Number of modes in the circuit.
        initial_state : list of tuple, optional
            The initial Fock state(s).
        stddev : float, optional
            Standard deviation for random parameter initialization.
        mean : float, optional
            Mean for random parameter initialization.
        config : dict, optional
            Configuration dictionary. Expected keys:
                - "num_modes": overrides num_modes.
                - "initial_state": overrides initial_state.
                - "memristor_positions": positions of memristors (as indices within the circuit; implementation follows later).
                - "initial_phases": list (or list of lists) of initial phases.
        """
        # Load parameters from config if provided.
        if config is not None:
            self.num_modes = config.get("num_modes", num_modes)
            self.initial_state = config.get("initial_state", initial_state)
            self.memristor_positions = config.get("memristor_positions", None)  # Implementation follows later.
            self.initial_phases = config.get("initial_phases", [])
        else:
            self.num_modes = num_modes
            self.initial_state = initial_state
            self.initial_phases = []
            self.memristor_positions = None
        
        self.stddev = stddev
        self.mean = mean
        
        # Calculate the number of photons from the initial state.
        self.num_photons = max(sum(state) for state in self.initial_state)
        # Define the initial state in the Fock basis.
        self.init_state = MemristorCircuit.fock_basis(self.initial_state, self.num_photons + 1, self.num_modes)[0]
        # Total number of free parameters; note that each gate uses two parameters.
        self.num_parameters = self.num_modes * (self.num_modes - 1)
        
        # Define the Strawberry Fields program and engine.
        self.program = sf.Program(self.num_modes)
        self.engine = sf.Engine("tf", backend_options={"cutoff_dim": self.num_photons + 1, "batch_size": None})
        
        # Setup parameters and the interferometer structure.
        self.setup_parameters()
        self.setup_structure()
        
        # Build the circuit using the provided initial phases if any.
        # If no phases are provided, the circuit will be built later with default/random values.
        if self.initial_phases:
            self.build_circuit(self.initial_phases)
        else:
            # Build a circuit with no explicit initial phases (could be later mapped with run()).
            self.build_circuit([])

    def setup_parameters(self):
        """
        Set up the Strawberry Fields parameter list for the interferometer.
        Each gate requires two parameters: theta and phi.
        """
        num_gates = self.num_parameters // 2
        self.sf_params = []
        para_names = ['theta', 'phi']
        for i in range(num_gates):
            param_names = ['{}_{}'.format(name, i) for name in para_names]
            self.sf_params.append(self.program.params(*param_names))
        self.sf_params = np.array(self.sf_params)

    def setup_structure(self):
        """
        Build the Clements universal interferometer structure.
        
        Note: In a later version, memristor positions will be read from the config file.
        """
        self.structure = []
        # Create pairs based on even-odd indexing.
        for _ in range(self.num_modes // 2):
            for i in range(2):
                for j in range(self.num_modes - 1):
                    if j % 2 == i:
                        self.structure.append((j, j + 1))
        if (self.num_modes % 2) != 0:
            for j in range(self.num_modes - 1):
                if j % 2 == 0:
                    self.structure.append((j, j + 1))
    
    def build_circuit(self, initial_phases):
        """
        Build the quantum circuit using the provided initial phases.
        
        Parameters
        ----------
        initial_phases : list
            Arbitrary long list of initial phases. This can be a flat list or a list of lists.
        
        Behavior
        --------
        - Prints out the size of the initial phases list and its compatibility with num_modes.
        - If initial_phases is a list of lists, prints "continuous variables detected" and takes the average of each inner list.
          (Note: one can implement sampling here as well later.)
        - Uses the processed phases to set the parameters of the circuit as before.
        """
        # Process the input phases.
        if initial_phases and isinstance(initial_phases[0], list):
            print("continuous variables detected")
            # Take the average of every inner list.
            # (In future, one might sample from these distributions instead.)
            processed_phases = [np.mean(phase) for phase in initial_phases]
        else:
            processed_phases = initial_phases
        
        # Print out the size and compatibility.
        print("Initial phases provided:", processed_phases)
        num_phase_values = len(processed_phases)
        if num_phase_values > 0 and num_phase_values % 2 != 0:
            raise ValueError("The number of phase values must be even (pairs for theta and phi for each gate).")
        num_gates_from_phases = num_phase_values // 2 if num_phase_values > 0 else 0
        print("Number of gates from initial phases:", num_gates_from_phases)
        print("Number of modes:", self.num_modes)
        
        if num_gates_from_phases > len(self.structure):
            print("Warning: More gates specified in initial phases than available in the circuit structure based on num_modes.")
        
        # If no phases were provided, do nothing further (the circuit can be run later with default mapping).
        if num_gates_from_phases == 0:
            return
        
        # Reshape the phases into an array of shape (num_gates, 2).
        phases_array = np.array(processed_phases).reshape(-1, 2)
        
        # Build the circuit: reinitialize the program context.
        with self.program.context as q:
            # Initialize the state.
            Ket(self.init_state) | q
            
            # Apply MZ gates using the interferometer structure.
            for p in range(num_gates_from_phases):
                if p < len(self.structure):
                    modes = self.structure[p]
                    # Apply the MZgate with provided theta and phi.
                    MZgate(phases_array[p, 0], phases_array[p, 1]) | (q[modes[0]], q[modes[1]])
                else:
                    # If more gates are provided than available positions, ignore the extra ones.
                    print(f"Extra gate at index {p} ignored due to insufficient memristor positions.")

    def run(self):
        """
        Run the quantum circuit.
        
        If the circuit was built with provided initial phases, those parameters will be used.
        Otherwise, default/random parameters may be mapped in a subsequent implementation.
        
        Returns
        -------
        numpy.ndarray
            The Fock basis probabilities after executing the circuit.
        """
        # Execute the program.
        state = self.engine.run(self.program).state
        probabilities = state.all_fock_probs()
        self.engine.reset()
        return probabilities

# # Example usage:
# if __name__ == "__main__":
#     # Example configuration (in practice, load from a config file)
#     config = {
#         "num_modes": 3,
#         "initial_state": [(1, 0, 0)],
#         # "memristor_positions": ...  # Positions of memristors provided by the config; implementation follows later.
#         # Flat list of phases for two gates: [theta_0, phi_0, theta_1, phi_1]
#         "initial_phases": [0.1, 0.2, 0.3, 0.4]
#         # Alternatively, for continuous variables one might provide:
#         # "initial_phases": [[0.1, 0.15, 0.2], [0.25, 0.2, 0.15], [0.3, 0.35, 0.4], [0.45, 0.4, 0.35]]
#     }
#     circuit = MemristorCircuit(config=config)
#     probs = circuit.run()
#     print("Fock probabilities:\n", probs)


class MemristorCircuit:
    def __init__(self, phase1, memristor_weight, phase3, encoded_phases):
        self.phase1 = phase1
        self.memristor_weight = memristor_weight
        self.phase3 = phase3
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_memristor_weight(self, memristor_weight):
        self.memristor_weight = memristor_weight

    def set_phase3(self, phase3):
        self.phase3 = phase3

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the quantum circuit with the given parameters.
        """
        circuit = sf.Program(3)
        with circuit.context as q:
            Vac     | q[0]
            Fock(1) | q[1]
            Vac     | q[2]

            # Input encoding MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.encoded_phases)           | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # First MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase1)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # Memristor (Second MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase3)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        return circuit
    

class MemristorMegaCircuit:
    def __init__(self, phase1, memristor_weight, phase3, phase4, memristor2_weight, phase6, encoded_phases):
        self.phase1 = phase1
        self.memristor_weight = memristor_weight
        self.phase3 = phase3
        self.phase4 = phase4
        self.memristor2_weight = memristor2_weight
        self.phase6 = phase6
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_memristor_weight(self, memristor_weight):
        self.memristor_weight = memristor_weight

    def set_phase3(self, phase3):
        self.phase3 = phase3
    
    def set_phase4(self, phase4):
        self.phase4 = phase4

    def set_memristor2_weight(self, memristor2_weight):
        self.memristor2_weight = memristor2_weight

    def set_phase6(self, phase6):
        self.phase6 = phase6

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the longer quantum circuit with the given parameters.
        """
        circuit = sf.Program(3)
        with circuit.context as q:
            Vac     | q[0]
            Fock(1) | q[1]
            Vac     | q[2]
        
            # Input encoding MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.encoded_phases)           | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
            # First MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase1)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
            # First Memristor (Second MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase3)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

            # Fourth MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase4)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
        
            # Second Memristor (Fifth MZI)
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.memristor2_weight)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
        
            # Sixth MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])
            Rgate(self.phase6)             | q[1]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[1])

        return circuit
    

class MemristorMegaBigCircuit:
    def __init__(self, phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, encoded_phases):
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3
        self.phase4 = phase4
        self.phase5 = phase5
        self.phase6 = phase6
        self.phase7 = phase7
        self.phase8 = phase8
        self.phase9 = phase9
        self.phase10 = phase10
        self.phase11 = phase11
        self.phase12 = phase12
        self.encoded_phases = encoded_phases

    def set_phase1(self, phase1):
        self.phase1 = phase1

    def set_phase2(self, phase2):
        self.phase2 = phase2

    def set_phase3(self, phase3):
        self.phase3 = phase3
    
    def set_phase4(self, phase4):
        self.phase4 = phase4

    def set_phase5(self, phase5):
        self.phase5 = phase5
    
    def set_phase6(self, phase6):
        self.phase6 = phase6

    def set_phase7(self, phase7):
        self.phase7 = phase7
    
    def set_phase8(self, phase8):
        self.phase8 = phase8

    def set_phase9(self, phase9):
        self.phase9 = phase9

    def set_phase10(self, phase10):
        self.phase10 = phase10
    
    def set_phase11(self, phase11):
        self.phase11 = phase11

    def set_phase12(self, phase12):
        self.phase12 = phase12

    def set_encoded_phases(self, encoded_phases):
        self.encoded_phases = encoded_phases

    def build_circuit(self):
        """
        Constructs the longer quantum circuit with the given parameters.
        """
        circuit = sf.Program(6)
        with circuit.context as q:
            Vac     | q[0]
            Vac     | q[1]
            Fock(1) | q[2]
            Vac     | q[3]
            Vac     | q[4]
            Vac     | q[5]
        
            # Input encoding MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.encoded_phases)           | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
        
            # First MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase1)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Second MZI
            BSgate(np.pi/4, np.pi/2) | (q[4], q[2])
            Rgate(self.phase2)             | q[4]
            BSgate(np.pi/4, np.pi/2) | (q[4], q[2])
        
            # Third MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase3)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Fourth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.phase4)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
        
            # Fith MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])
            Rgate(self.phase5)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])

            # Sixth MZI
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.phase6)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Seventh MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])
            Rgate(self.phase7)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])

            # Eith MZI
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])
            Rgate(self.phase8)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[0], q[2])

            # Ninth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])
            Rgate(self.phase9)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[3])

            # Tenth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])
            Rgate(self.phase10)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[5])

            # Eleventh MZI
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])
            Rgate(self.phase11)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[1], q[2])

            # Twelth MZI
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])
            Rgate(self.phase12)             | q[2]
            BSgate(np.pi/4, np.pi/2) | (q[2], q[4])

        return circuit