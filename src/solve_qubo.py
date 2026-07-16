from dwave.samplers import SimulatedAnnealingSampler  # ou "from neal import ..." selon l'etape 2

def solve_qubo(bqm, num_reads=1000):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    return sampleset.first