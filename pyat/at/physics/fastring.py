"""
Functions relating to fast_ring
"""
import numpy
import functools
from at.lattice import get_cells, checktype, checkname
from at.lattice import RFCavity, Marker, Drift
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

    def pack(counter, rslice, orb_beg, orb_end):
        ibeg = ring.index(mark)
        fast = rslice[:ibeg]
        lin_elem_rad = gen_m66_elem(rslice[ibeg + 1:-1], orb_beg, orb_end)
        lin_elem_rad.FamName = lin_elem_rad.FamName + '_' + str(counter)
        fast.append(lin_elem_rad)
        return fast

    def packrad(counter, rslice, orb_beg, orb_end):
        fastrad = pack(counter, rslice, orb_beg, orb_end)
        qd_elem = gen_quantdiff_elem(rslice, orbit=orb_beg)
        qd_elem.FamName = qd_elem.FamName + '_' + str(counter)
        fastrad.append(qd_elem)
        return fastrad

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

    fast = [pack(counter, rg.radiation_off(copy=True), orb1, orb2)
            for (counter, rg), orb1, orb2
            in zip(enumerate(all_rings), orbit4[0::2], orbit4[1::2])]

    fastrad = [packrad(counter, rg, orb1, orb2)
            for (counter, rg), orb1, orb2
            in zip(enumerate(all_rings), orbit6[0::2], orbit6[1::2])]

    resnorad = functools.reduce(lambda x, y: x + y, fast)
    resrad = functools.reduce(lambda x, y: x + y, fastrad)

    resnorad.append(detuning_elem)
    resrad.append(detuning_elem_rad)

    return resnorad, resrad
