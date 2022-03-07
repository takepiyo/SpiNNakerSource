import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
from pyNN.random import RandomDistribution
import matplotlib.pyplot as plt

simtime = 5 * 1000
sim.setup(timestep=1.0)
n_neurons = 100
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, n_neurons)

pop = sim.Population(n_neurons, sim.IF_curr_exp(), label="pop")
input = sim.Population(1, sim.SpikeSourceArray(spike_times=[0]), label="input")

input_to_pop_weight = 5
input_to_pop_delay = 1
pop_to_pop_weight = 5
pop_to_pop_delay = RandomDistribution('uniform', low=1.0, high=15.0)

sim.Projection(input, pop, sim.FromListConnector(
    [(0, 0, input_to_pop_weight, input_to_pop_delay)], column_names=('weight', 'delay')))

sim.Projection(pop, pop, sim.FromListConnector([(i, (i + 1) % n_neurons, pop_to_pop_weight, pop_to_pop_delay.next()) for i in range(
    n_neurons)], column_names=("weight", "delay")))
pop.record("spikes")

sim.run(simtime)

spikes = pop.get_data("spikes")

figure_filename = "results.png"
figure = Figure(
    Panel(spikes.segments[0].spiketrains, yticks=True,
          markersize=0.5, xlim=(0, simtime)),
    title="Task 2-2",
    annotations="Simulated with {}".format(sim.name())
)
figure.save(figure_filename)
plt.show()
sim.end()
