import strawberryfields as sf
from strawberryfields.ops import Interferometer, Fock, Vac

# Number of modes
M = 3

# Your Clements parameters:
#  - theta: array of beam-splitter angles (size = M(M-1)/2)
#  - phi:   array of internal phases      (size = M(M-1)/2)
#  - varphi: array of final local rotations (size = M)
theta  = [...]  # transmissivities
phi    = [...]  # internal MZI phases
varphi = [...]  # final mode phases

# Build the program
prog = sf.Program(M)
with prog.context as q:
    # Prepare your Fock inputs
    for i, n in enumerate([1, 0, 1]):
        (Fock(1) if n else Vac) | q[i]

    # Place the full interferometer in one go
    Interferometer(
        theta, phi, varphi,
        mesh="rectangular",           # Clements layout
        drop_identity=False           # keep all phases
    ) | q  # :contentReference[oaicite:0]{index=0}

# Run it on a Fock backend
eng = sf.Engine("fock", backend_options={"cutoff_dim": 2})
result = eng.run(prog)
print(result.state.all_fock_probs())