import spynnaker8 as p
from spynnaker.pyNN.models.neuron.builds import IFCondExpStoc
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import numpy as np
import matplotlib.pyplot as plt
import random

p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(IFCondExpStoc, 100)

exc = p.Population(10, IFCondExpStoc, label="exc")
inh = p.Population(1, IFCondExpStoc, label="inh")

input = p.Population(2, p.SpikeSourceArray(
    [[5.0, 25.0], [15.0, 35.0]]), label="input")

p.Projection(input, exc, p.FromListConnector([(0, 0, 15.0, 1.0)]))
p.Projection(input, exc, p.FromListConnector([(1, 5, 15.0, 1.0)]))

p.Projection(exc, inh, p.AllToAllConnector(), synapse_type=p.StaticSynapse(
    weight=10.0, delay=1.0), receptor_type="excitatory")
p.Projection(exc, inh, p.AllToAllConnector(), synapse_type=p.StaticSynapse(
    weight=10.0, delay=1.0), receptor_type="inhibitory")

exc.record(["v", "spikes"])

p.run(40.0)

spikes = exc.get_data('spikes')

figure_filename = "results.png"
figure = Figure(
    Panel(spikes.segments[0].spiketrains, yticks=True, xticks=True,
          markersize=0.5, xlim=(0, 40.0)),
    title="Task WTA",
    annotations="Simulated with {}".format(p.name())
)
figure.save(figure_filename)
plt.show()
p.end()
