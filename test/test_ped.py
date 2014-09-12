import unittest
import sys
import os
sys.path.append(os.path.join('..', 'src'))
import numpy as np
import anhima.ped

# build parents with combinations of genotypes
href_parent = np.zeros((1,1,2))
halt_parent = np.ones((1,1,2),dtype='int')
het_parent = np.array([0,1],dtype='int').reshape((1,1,2))

# give progeny 4 genotypes, 00, 01, 11, 12
progeny = np.array([[0,0], [0,1], [1,1], [-1,-1]]).reshape(1,-1,2)

class test_mendelian_error(unittest.TestCase):
    def test_test(self):
        self.assertTrue(True)

    def test_functionExists(self):
        self.assertTrue('is_non_mendelian_diploid' in dir(anhima.ped))

    def test_homref_homref(self):
      parents = np.hstack([href_parent, href_parent]).astype(int)
      non_mendelian = anhima.ped.is_non_mendelian_diploid(
          parental_genotypes = parents,
          progeny_genotypes  = progeny
      )
      print non_mendelian[0]
      self.assertTrue(np.array_equal(non_mendelian[0], [False, True, True, False]))

    def test_homref_het(self):
         parents = np.hstack([href_parent, het_parent]).astype(int)
         non_mendelian = anhima.ped.is_non_mendelian_diploid(
             parental_genotypes = parents,
             progeny_genotypes  = progeny,
         )
         print non_mendelian[0]
         self.assertTrue(np.array_equal(non_mendelian[0], [False, False, True, False]))

    def test_homref_homalt(self):
         parents = np.hstack([halt_parent, href_parent]).astype(int)
         non_mendelian = anhima.ped.is_non_mendelian_diploid(
             parental_genotypes = parents,
             progeny_genotypes  = progeny,
         )
         print non_mendelian[0]
         self.assertTrue(np.array_equal(non_mendelian[0], [True, False, True, False]))

    def test_homalt_het(self):
         parents = np.hstack([halt_parent, het_parent]).astype(int)
         non_mendelian = anhima.ped.is_non_mendelian_diploid(
             parental_genotypes = parents,
             progeny_genotypes  = progeny,
         )
         print non_mendelian[0]
         self.assertTrue(np.array_equal(non_mendelian[0], [True, False, False, False]))

    def test_homalt_homalt(self):
         parents = np.hstack([halt_parent, halt_parent]).astype(int)
         non_mendelian = anhima.ped.is_non_mendelian_diploid(
             parental_genotypes = parents,
             progeny_genotypes  = progeny,
         )
         print non_mendelian[0]
         self.assertTrue(np.array_equal(non_mendelian[0], [True, True, False, False]))

    def test_error_multiallelic(self):
         parents = np.hstack([href_parent, halt_parent]).astype(int)
         progeny_multi = np.array([[0,2], [0,1], [1,2], [-1,-1]]).reshape(1,-1,2)

         self.assertRaises(
             AssertionError, 
             anhima.ped.is_non_mendelian_diploid,
             parental_genotypes = parents,
             progeny_genotypes  = progeny_multi,
         )
