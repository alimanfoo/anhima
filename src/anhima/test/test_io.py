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

        # read in file
        with open(path.name) as f:
            content = f.readlines()

        self.assertEqual(n_variants, len(content))

        # test that content is good...by taking first line
        line = content[0].split("\t")
        self.assertEquals(n_samples + 4, len(line))
        self.assertEquals('_'.join(line[0:4]), '_'.join(['0', 'snp0', '0.0',
                                                         str(pos[0])]))
    # test that we can load from hdf5 and create tped

    # test that we can write to hdf5 ok
