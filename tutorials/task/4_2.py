import spynnaker8 as p
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel, DataTable
import numpy as np
import matplotlib.pyplot as plt
import random

p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 200)

train_time = 500
test_time = 0
simtime = train_time + test_time

pop_size = 100

pre_stim_time = [[i, i + 200] for i in range(pop_size)]
# pre_stim_time.extend(map(lambda x: x + train_time, [20, 40, 60, 80, 100]))
post_stim_time = [[50] for i in range(pop_size)]

pre_pop = p.Population(pop_size, p.IF_curr_exp, label="pre")
post_pop = p.Population(pop_size, p.IF_curr_exp, label="post")
pre_input_pop = p.Population(
    pop_size, p.SpikeSourceArray(pre_stim_time), label="input_pre")
post_input_pop = p.Population(
    pop_size, p.SpikeSourceArray(post_stim_time), label="input_post")

p.Projection(pre_input_pop, pre_pop, p.OneToOneConnector(),
             synapse_type=p.StaticSynapse(5.0, 1.0))
p.Projection(post_input_pop, post_pop, p.OneToOneConnector(),
             synapse_type=p.StaticSynapse(5.0, 1.0))

stdp_model = p.STDPMechanism(timing_dependence=p.SpikePairRule(tau_plus=50.0, tau_minus=20.0, A_plus=0.2, A_minus=0.02),
                             weight_dependence=p.AdditiveWeightDependence(w_min=0.0, w_max=1.0), weight=0.5, delay=1.0)

stdp_proj = p.Projection(
    pre_pop, post_pop, p.OneToOneConnector(), synapse_type=stdp_model)

pre_pop.record(["v", "spikes"])
post_pop.record(["v", "spikes"])

p.run(simtime)

weights = stdp_proj.get(["weight"], "list")
final_weights = np.array([w[-1] for w in weights])
deltas = np.arange(-50, 50, -1)
plasticity_data = DataTable(deltas, final_weights)

print("Weights:{}".format(stdp_proj.get('weight', 'list')))

pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')

figure_filename = "results.png"
figure = Figure(
    Panel(pre_spikes.segments[0].spiketrains, yticks=True, xticks=True,
          markersize=0.5, xlim=(0, simtime)),
    Panel(post_spikes.segments[0].spiketrains, yticks=True, xticks=True,
          markersize=0.5, xlim=(0, simtime)),
    Panel(plasticity_data, xticks=True, yticks=True, xlim=(-50, 50), ylim=(0.9 * final_weights.min(),
          1.1 * final_weights.max()), xlabel="t_post - t_pre (ms)", ylabel="Final weight (nA)"),
    title="Task 5",
    annotations="Simulated with {}".format(p.name())
)
figure.save(figure_filename)
plt.show()
p.end()
