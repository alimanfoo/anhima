import timeit

# simulate genotypes 

command = [
    'import imp;',
    'import os;',
    'import anhima;',
    'ah = imp.load_source("anhima", os.path.join("src", "anhima", "ped.py"));',
    'import scipy.stats;',
    'n_samples = 10;',
    'n_variants = 10**6;',
    'ploidy = 2;',
    'af_dist = scipy.stats.beta(a=.9, b=.1);',
    'p_missing = .1;',
    'parents = anhima.sim.simulate_biallelic_genotypes(n_variants, 2, af_dist, p_missing=0.1, ploidy=2);',
    'progeny = anhima.sim.simulate_biallelic_genotypes(n_variants, n_samples, af_dist, p_missing=0.1, ploidy=2);',
    'non_mendelian = ah.anhima.is_non_mendelian_diploid(parental_genotypes = parents, progeny_genotypes = progeny);'
    ]

print timeit.timeit(" ".join(command), number=1)
