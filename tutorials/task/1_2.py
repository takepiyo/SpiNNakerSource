import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

pop = sim.Population(2, sim.IF_curr_exp(tau_syn_E=1.0), label="pop")
input = sim.Population(2, sim.SpikeSourceArray([[0.0], [1.0]]), label="input")

input_proj = sim.Projection(input, pop, sim.OneToOneConnector(
), synapse_type=sim.StaticSynapse(weight=5.0, delay=2))

pop.record(["spikes", "v"])

simtime = 10
sim.run(simtime)

v = pop.get_data("v")
spikes = pop.get_data("spikes")

print(v.segments[0].filter(name="v"))
print(spikes.segments[0].spiketrains)

figure_filename = "results.png"
figure = Figure(
    Panel(spikes.segments[0].spiketrains, yticks=True,
          markersize=0.2, xlim=(0, simtime)),
    Panel(v.segments[0].filter(name="v")[0], ylabel="Membrane potential (mV)", data_labels=[
          pop.label], yticks=True, xlim=(0, simtime)),
    title="Task 1-1",
    annotations="Simulated with {}".format(sim.name())
)
figure.save(figure_filename)

sim.end()
