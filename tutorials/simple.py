import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt

sim.setup()
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

pop_1 = sim.Population(1, sim.IF_curr_exp(), label="pop_1")
input = sim.Population(1, sim.SpikeSourceArray(spike_times=[0]), label="input")
input_proj = sim.Projection(input, pop_1, sim.OneToOneConnector(
), synapse_type=sim.StaticSynapse(weight=5, delay=1))

pop_1.record(["spikes", "v"])
simtime = 10
sim.run(simtime)

neo = pop_1.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
print(spikes)
v = neo.segments[0].filter(name="v")[0]
print(v)
sim.end()

fig = plot.Figure(
    # plot voltage for first ([0]) neuron
    plot.Panel(v, ylabel="Membrane potential (mv)", data_labels=[
               pop_1.label], yticks=True, xlim=(0, simtime)),
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)), title="Simple Example", annotations="Simulated with {}".format(sim.name())
)
# plt.show()
fig.save("{}.png".format(__file__).replace(".py", ""))
