import spynnaker8 as p
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import numpy as np
import matplotlib.pyplot as plt

p.setup(timestep=0.1)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 64)
p.set_number_of_neurons_per_core(p.SpikeSourcePoisson, 64)

n_neurons = 500
n_exc = int(n_neurons * 0.8)
n_inh = int(n_neurons * 0.2)
poisson_rate = 1000
rng = p.NumpyRNG(0)

input_exc_pop = p.Population(n_exc, p.SpikeSourcePoisson(
    rate=poisson_rate), label="input_exc", additional_parameters={"seed": 1})
input_inh_pop = p.Population(n_inh, p.SpikeSourcePoisson(
    rate=poisson_rate), label="input_inh", additional_parameters={"seed": 2})
exc_pop = p.Population(n_exc, p.IF_curr_exp(), label="exc")
inh_pop = p.Population(n_inh, p.IF_curr_exp(), label="inh")
p.Projection(input_exc_pop, exc_pop, p.OneToOneConnector(),
             p.StaticSynapse(0.1, 1.0), receptor_type="excitatory")
p.Projection(input_inh_pop, inh_pop, p.OneToOneConnector(),
             p.StaticSynapse(0.1, 1.0), receptor_type="excitatory")
synapse = p.StaticSynapse(
    RandomDistribution("normal_clipped", mu=0.11, sigma=0.1, low=0.0, high=np.inf, rng=rng), RandomDistribution("normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=1.6, rng=rng))
p.Projection(exc_pop, inh_pop, p.FixedProbabilityConnector(
    0.1, rng=rng), synapse, receptor_type="excitatory")
p.Projection(exc_pop, exc_pop, p.FixedProbabilityConnector(
    0.1, rng=rng), synapse, receptor_type="excitatory")
synapse = p.StaticSynapse(
    RandomDistribution("normal_clipped", mu=-0.44, sigma=0.1,
                       low=-np.inf, high=0.0, rng=rng),
    RandomDistribution("normal_clipped", mu=0.75, sigma=0.375, low=1.0, high=1.6, rng=rng))
p.Projection(inh_pop, exc_pop, p.FixedProbabilityConnector(
    0.1, rng=rng), synapse, receptor_type="inhibitory")
p.Projection(inh_pop, inh_pop, p.FixedProbabilityConnector(
    0.1, rng=rng), synapse, receptor_type="inhibitory")

exc_pop.initialize(v=RandomDistribution(
    "uniform", low=-65.0, high=-55.0, rng=rng))
inh_pop.initialize(v=RandomDistribution(
    "uniform", low=-65.0, high=-55.0, rng=rng))

exc_pop.record("spikes")

simtime = 1000
p.run(simtime)

spikes = exc_pop.get_data("spikes")

figure_filename = "results.png"
figure = Figure(
    Panel(spikes.segments[0].spiketrains, yticks=True,
          markersize=1.0, xlim=(0, simtime)),
    title="Task 3-2",
    annotations="Simulated with {}".format(p.name())
)
figure.save(figure_filename)
plt.show()
p.end()
