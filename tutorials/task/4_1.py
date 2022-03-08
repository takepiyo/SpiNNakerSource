import spynnaker8 as p
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import numpy as np
import matplotlib.pyplot as plt
import random

p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 100)

simtime = 1200

pre_stim_time = [10 * i for i in range(100)]
pre_stim_time.extend([1050, 1075, 1100, 1125, 1150])
post_stim_time = [10 * i + random.randint(-1, 4) for i in range(100)]

pre_pop = p.Population(1, p.IF_curr_exp, label="pre")
post_pop = p.Population(1, p.IF_curr_exp, label="post")
input = p.Population(2, p.SpikeSourceArray(
    [pre_stim_time, post_stim_time]), label="input")

p.Projection(input, pre_pop, p.FromListConnector([(0, 0, 5.0, 1.0)]))
p.Projection(input, post_pop, p.FromListConnector([(1, 0, 5.0, 1.0)]))

stdp_model = p.STDPMechanism(timing_dependence=p.SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.02, A_minus=0.02),
                             weight_dependence=p.AdditiveWeightDependence(w_min=0.0, w_max=5.0))

stdp_proj = p.Projection(
    pre_pop, post_pop, p.OneToOneConnector(), synapse_type=stdp_model)

pre_pop.record(["v", "spikes"])
post_pop.record(["v", "spikes"])

p.run(simtime)

print("Weights:{}".format(stdp_proj.get('weight', 'list')))

pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')

figure_filename = "results.png"
figure = Figure(
    Panel(pre_spikes.segments[0].spiketrains, yticks=True, xticks=True,
          markersize=0.5, xlim=(0, simtime)),
    Panel(post_spikes.segments[0].spiketrains, yticks=True, xticks=True,
          markersize=0.5, xlim=(0, simtime)),
    title="Task 4-1",
    annotations="Simulated with {}".format(p.name())
)
figure.save(figure_filename)
plt.show()
p.end()
