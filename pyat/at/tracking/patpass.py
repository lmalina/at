"""
Simple parallelisation of atpass() using multiprocessing.
"""
import multiprocessing
from at.tracking import atpass
from at.lattice import uint32_refpts
from sys import platform
import numpy


__all__ = ['patpass']

if platform.startswith('linux'):
    globring = None

    def _atpass_one(args):
        return atpass(globring, *args)

    def _atpass(ring, r_in, nturns, refs, pool_size=None):
        global globring
        globring = ring
        args = [(r_in[:, i], nturns, refs) for i in range(r_in.shape[1])]
        with multiprocessing.Pool(pool_size) as pool:
            results = pool.map(_atpass_one, args)
        vals = numpy.concatenate(results, axis=1)
        globring = None
        return vals
else:
    def _atpass(ring, r_in, nturns, refs, pool_size=None):
        args = [(ring, r_in[:, i], nturns, refs) for i in range(r_in.shape[1])]
        with multiprocessing.Pool(pool_size) as pool:
            results = pool.starmap(atpass, args)
        return numpy.concatenate(results, axis=1)


def patpass(ring, r_in, nturns, refpts=None, reuse=True, pool_size=None):
    """
    Simple parallel implementation of atpass().  If more than one particle
    is supplied, use multiprocessing to run each particle in a separate
    process.

    INPUT:
        ring            lattice description
        r_in:           6xN array: input coordinates of N particles
        nturns:         number of passes through the lattice line
        refpts          elements at which data is returned. It can be:
                        1) an integer in the range [-len(ring), len(ring)-1]
                           selecting the element according to python indexing
                           rules. As a special case, len(ring) is allowed and
                           refers to the end of the last element,
                        2) an ordered list of such integers without duplicates,
                        3) a numpy array of booleans of maximum length
                           len(ring)+1, where selected elements are True.
                        Defaults to None, meaning no refpts, equivelent to
                        passing an empty array for calculation purposes.
    """
    if not reuse:
        raise ValueError('patpass does not support altering lattices')
    if refpts is None:
        refpts = len(ring)
    refs = uint32_refpts(refpts, len(ring))
    return _atpass(ring, r_in, nturns, refs, pool_size=pool_size)
