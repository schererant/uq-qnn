import perceval as pcvl
from perceval.algorithm import Analyzer, Sampler


class PhotonicMachineLearning:
    def __init__(self, size):
        size = size # M for M X M circuit
        circuit = None

    def build_circuit(size, phases, circuit_type="Clements") -> pcvl.Circuit:
        
        if circuit_type == "Clements":
            """
            Build the Clements circuit layout.
            """
            
            # Check if the number of phases matches the expected number
            expected_num_phases = size * (size - 1) // 2
            if len(phases) != expected_num_phases:
                raise ValueError(f"Expected {expected_num_phases} phases, but got {len(phases)}")
            
            # Initialize the circuit
            circuit = pcvl.Circuit(size)
            param_idx = 0

            # # Add a single-photon source to the circuit
            # source = pcvl.Source(emission_probability=1.0, indistinguishability=0.95)
            # circuit.add_source(source, [2])  # Adding the source to mode 2

            # Implement the rectangular mesh (Clements layout)
            for layer in range(size):
                start_idx = layer % 2  # Alternating between even and odd layers
                
                for i in range(start_idx, size - 1, 2):
                    # Add a Mach-Zehnder interferometer
                    circuit.add((i, i+1), pcvl.BS())
                    circuit.add((i+1,), pcvl.PS(phi=phases[param_idx]))
                    circuit.add((i, i+1), pcvl.BS())
                    param_idx += 1
                    
                    if param_idx >= len(phases):
                        break
                    
    def _sim_circuit(self, input_fock, backend="SLOS", noise_model=None):
        """
        Simulate the circuit with the given input Fock state.
        """
        # 2) Wrap it in a Processor on the Fock backend
        proc = pcvl.Processor(backend, circuit)
        proc.check_min_detected_photons_filter = 1

        # 3) Inject one photon into mode 1 (BasicState([0,1,0]))
        proc.with_input(input_fock)

        # 3a) Add a noise model to the processor (example: photon loss and phase noise)
        if noise_model:
            proc.with_noise(noise_model)

        # 4) Sample 1000 shots
        sampler = Sampler(proc)
        counts = sampler.sample_count(100)
        probs = sampler.probs(100)
        
        return counts['results'], probs['results']
                    
                    
    