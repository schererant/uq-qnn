import strawberryfields as sf
from strawberryfields.ops import MZgate, Vac, Fock


# Parameter 

circuit = sf.Program(3)
with circuit.context as q:
    Vac     | q[0]
    Fock(3) | q[1]
    Vac     | q[2]

    MZgate(phi_in=0, phi_ex=0)



eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 4})
results = eng.run(circuit)
prob = results.state.all_fock_probs()

# Create print for results and prob
print("Results: ", results)
print("Prob: ", prob)

# sf.plot.generate_fock_chart(results.state, [0,1,2], 3)