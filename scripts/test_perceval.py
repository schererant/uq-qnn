import perceval as pcvl
from perceval.algorithm import Analyzer, Sampler
# from perceval.backend import NoiseModel

# Fixed parameters
SIZE = 6
INPUT_FOCK = pcvl.BasicState([0, 0, 1, 0, 0, 0])
PHASES = [0] * (SIZE * (SIZE - 1) // 2)  # All phases set to 0
BACKEND = "SLOS"

# Build the circuit
circuit = pcvl.Circuit(SIZE)
param_idx = 0

#TODO: Add source
#TODO: Add noise

# Add a single-photon source to the circuit
source = pcvl.Source(emission_probability=1.0, indistinguishability=0.95)
circuit.add_source(source, [2])  # Adding the source to mode 2

# Implement the rectangular mesh (Clements layout)
for layer in range(SIZE):
    start_idx = layer % 2  # Alternating between even and odd layers
    
    for i in range(start_idx, SIZE - 1, 2):
        # Add a Mach-Zehnder interferometer
        circuit.add((i, i+1), pcvl.BS())
        circuit.add((i+1,), pcvl.PS(phi=PHASES[param_idx]))
        circuit.add((i, i+1), pcvl.BS())
        param_idx += 1
        
        if param_idx >= len(PHASES):
            break

print("Circuit:", circuit)

noise_model = pcvl.NoiseModel(brightness=0.3, g2=0.05)

# 2) Wrap it in a Processor on the Fock backend
source = pcvl.Source(brightness=0.3, purity=0.95, purity_model="distinguishable")
proc = pcvl.Processor("SLOS", circuit)
proc.check_min_detected_photons_filter = 1

# 3) Inject one photon into mode 1 (BasicState([0,1,0]))
proc.with_input(INPUT_FOCK)

# 3a) Add a noise model to the processor (example: photon loss and phase noise)
# noise = pcvl.NoiseModel(error_rates={'photon_loss': 0.1, 'phase_noise': 0.05})
# proc.with_noise(noise)

# 4) Sample 1000 shots
sampler = Sampler(proc)
counts = sampler.sample_count(100)
probs = sampler.probs(100)
print("Counts:", counts['results'])
print("Probabilities:", probs['results'])

# 5) Analyze the results
analyzer = Analyzer(proc, INPUT_FOCK)
analyzer.analyze(counts['results'], probs['results'])
