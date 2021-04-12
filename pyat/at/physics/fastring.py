"""
Functions relating to fast_ring
"""
import numpy
import functools
from at.lattice import uint32_refpts, get_refpts, get_cells
from at.lattice import checktype, checkname, RFCavity, Marker, Drift
from at.physics import gen_m66_elem, gen_detuning_elem, gen_quantdiff_elem

__all__ = ['fast_ring']


def fast_ring(ring, split_inds=None):

    def rearrange(ring, marker, split_inds=None):
        limits = [0] + list(ring.uint32_refpts(split_inds)) + [len(ring) + 1]
        all_rings = [ring[i1:i2] for i1, i2 in zip(limits[:-1], limits[1:])]

        for slice in all_rings:
            # replace cavity with length > 0 with drift
            # set cavity length to 0 and move to start
            cavpts = ring.uint32_refpts(get_cells(ring, checktype(RFCavity)))
            cavities = []
            for cavindex in reversed(cavpts):
                cavity = slice.pop(cavindex)
                if cavity.Length != 0:
                    slice.insert(cavindex, Drift('CavDrift', cavity.Length))
                    cavity.Length = 0.0
                cavities.append(cavity)

            # merge all cavities with the same frequency
            freqs = numpy.array([cav.Frequency for cav in cavities])
            volts = numpy.array([cav.Voltage for cav in cavities])
            _, iu, iv = numpy.unique(freqs, return_index=True,
                                     return_inverse=True)

            slice.insert(0, marker)
            for ires, icav in enumerate(iu):
                cav = cavities[icav]
                cav.Voltage = sum([volts[iv == ires]])
                slice.insert(0, cav)
            slice.append(marker)

        return all_rings

    def pack(counter, slice, orb_beg, orb_end):
        ibeg = ring.index(mark)
        fastrad = slice[:ibeg]
        lin_elem_rad = gen_m66_elem(slice[ibeg + 1:-1], orb_beg, orb_end)
        lin_elem_rad.FamName = lin_elem_rad.FamName + '_' + str(counter)
        fastrad.append(lin_elem_rad)

        return fastrad

    def pack1(counter, ring_slice):
        ibeg = ring.index(mark)
        fastrad = ring_slice[:ibeg]
        fast=fastrad.radiation_off(copy=True)
        ring_slice_rad = ring_slice[ibeg+1:-1]
        ring_slice = ring_slice_rad.radiation_off(copy=True)

        lin_elem = gen_m66_elem(ring_slice, orbit4[2*counter],
                                orbit4[2*counter+1])
        lin_elem.FamName = lin_elem.FamName + '_' + str(counter)

        qd_elem = gen_quantdiff_elem(ring_slice_rad, orbit=orbit6[2*counter])
        qd_elem.FamName = qd_elem.FamName + '_' + str(counter)
        lin_elem_rad = gen_m66_elem(ring_slice_rad, orbit6[2*counter],
                                    orbit6[2*counter+1])
        lin_elem_rad.FamName = lin_elem_rad.FamName + '_' + str(counter)

        fast.append(lin_elem)
        fastrad.append(lin_elem_rad)
        fastrad.append(qd_elem)

        return fast, fastrad

    if not ring.radiation:
        ring = ring.radiation_on(copy=True)

    mark = Marker('')
    all_rings = rearrange(ring, mark, split_inds=split_inds)
    ringrad = functools.reduce(lambda x, y: x + y, all_rings)
    ringnorad = ringrad.radiation_off(copy=True)

    markers = get_cells(ringrad, lambda elem: elem is mark)

    _, orbit4 = ringnorad.find_sync_orbit(dct=0.0, refpts=markers)
    _, orbit6 = ringrad.find_orbit6(refpts=markers)

    detuning_elem = gen_detuning_elem(ringnorad, orbit4[-1])
    detuning_elem_rad = detuning_elem.deepcopy()
    detuning_elem_rad.T1 = -orbit6[-1]
    detuning_elem_rad.T2 = orbit6[-1]

    fast, fastrad = zip(*(pack(counter, ring_slice) for counter, ring_slice in enumerate(all_rings)))

    resnorad = functools.reduce(lambda x, y: x + y, fast)
    resrad = functools.reduce(lambda x, y: x + y, fastrad)

    resnorad.append(detuning_elem)
    resrad.append(detuning_elem_rad)

    return resnorad, resrad
