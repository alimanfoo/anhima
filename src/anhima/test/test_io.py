from __future__ import division, print_function, unicode_literals, \
    absolute_import


import unittest
import numpy as np
import os
import anhima.sim
import anhima.io
import scipy.stats
import tempfile

n_variants = 1000
ref = np.random.choice(['A', 'C', 'G', 'T'], n_variants)
alt = np.random.choice(['A', 'C', 'G', 'T'], n_variants)
pos = np.random.choice(range(n_variants*100), n_variants, False)
pos.sort()

# simulate genotypes
n_samples = 100
ploidy = 2
af_dist = scipy.stats.beta(a=.4, b=.6)
p_missing = .1
genotypes = anhima.sim.simulate_biallelic_genotypes(n_variants,
                                                    n_samples, af_dist,
                                                    p_missing, ploidy)


class TestTped(unittest.TestCase):

    def test_file_created(self):

        path = tempfile.NamedTemporaryFile(delete=False)
        print(path.name)
        anhima.io.save_tped(path.name, genotypes, ref, alt, pos)
        self.assertTrue(os.path.isfile(path.name))

        # need to count the rows
        num_lines = sum(1 for line in open(path.name))
        self.assertEqual(n_variants, num_lines)
