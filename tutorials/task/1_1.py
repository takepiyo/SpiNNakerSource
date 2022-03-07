from cProfile import label
import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt

sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

pop = sim.Population(2, sim.IF_curr_exp(), label="pop")
input = sim.Population(2, sim.SpikeSourceArray([[0.0], [1.0]]), label="input")

input_proj = sim.Projection(input, pop, sim.OneToOneConnector(
), synapse_type=sim.StaticSynapse(weight=5.0, delay=2))

pop.record(["spikes", "v"])
input.record(["spikes"])

simtime = 10
sim.run(simtime)

neo_pop = pop.get_data(variables=["spikes", "v"])
neo_input = input.get_data(variables=["spikes"])

# v_pop = [neo_pop.segments[0].filter(name="v")[0] for i in range(2)]
# v_input = [neo_input.segments[i].spiketrains for i in range(2)]

print(neo_pop.segments[0].filter(name="v"))
print(neo_pop.segments[0].spiketrains)
print(neo_input.segments[0].spiketrains)
