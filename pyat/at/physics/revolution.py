from ..lattice import Lattice, check_radiation, get_s_pos
from ..lattice import set_cavity, RFMode, get_rf_frequency
from ..lattice import DConstant, AtError
from ..lattice.constants import clight, e_mass
from ..tracking import lattice_pass
from .orbit import find_orbit4
from enum import Enum
import numpy
import math

__all__ = ['get_mcf', 'get_slip_factor', 'get_revolution_frequency',
           'set_rf_frequency', 'FRFMethod']


class FRFMethod(Enum):
    ANALYTIC = 'analytic'
    TRACKING = 'tracking'


@check_radiation(False)
def get_mcf(ring, dp=0.0, keep_lattice=False, **kwargs):
    """Compute the momentum compaction factor

    PARAMETERS
        ring            lattice description (radiation must be OFF)

    KEYWORDS
        dp=0.0          momentum deviation.
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        dp_step=1.0E-6  momentum deviation used for differentiation
    """
    dp_step = kwargs.pop('DPStep', DConstant.DPStep)
    fp_a, _ = find_orbit4(ring, dp=dp - 0.5*dp_step, keep_lattice=keep_lattice)
    fp_b, _ = find_orbit4(ring, dp=dp + 0.5*dp_step, keep_lattice=True)
    fp = numpy.stack((fp_a, fp_b),
                     axis=0).T  # generate a Fortran contiguous array
    b = numpy.squeeze(lattice_pass(ring, fp, keep_lattice=True), axis=(2, 3))
    ring_length = get_s_pos(ring, len(ring))
    alphac = (b[5, 1] - b[5, 0]) / dp_step / ring_length[0]
    return alphac


def get_slip_factor(ring, **kwargs):
    """Compute the slip factor

    PARAMETERS
        ring            lattice description (radiation must be OFF)

    KEYWORDS
        dp=0.0          momentum deviation.
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        dp_step=1.0E-6  momentum deviation used for differentiation
    """
    gamma = ring.gamma
    etac = (1.0/gamma/gamma - get_mcf(ring, **kwargs))
    return etac


def get_revolution_frequency(ring, dp=None, dct=None, **kwargs):
    """Compute the revolution frequency of the full ring [Hz]

    PARAMETERS
        ring            lattice description

    KEYWORDS
        dp=0.0          momentum deviation.
        dct=0.0         Path length deviation
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        dp_step=1.0E-6  momentum deviation used for differentiation
    """
    gamma = ring.gamma
    beta = math.sqrt(1.0 - 1.0 / gamma / gamma)
    frev = beta * clight / ring.circumference
    if dct is not None:
        frev -= frev * frev / clight * ring.periodicity * dct
    elif dp is not None:
        rnorad = ring.radiation_off(copy=True) if ring.radiation else ring
        etac = get_slip_factor(rnorad, **kwargs)
        frev += frev * etac * dp
    return frev


def set_rf_frequency(ring, frequency=None, dp=None, dct=None, cavpts=None,
                     copy=False, rfmode=RFMode.FUNDAMENTAL,
                     method=FRFMethod.ANALYTIC, df=1.0e-3, bpms=None, **kwargs):
    """Set the RF voltage

    PARAMETERS
        ring                lattice description
        frequency           RF frequency [Hz]

    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
        dp=0.0              momentum deviation used for analytic method
        dct=0.0             Path length deviation used for analytic method
        copy=False          If True, returns a shallow copy of ring with new
                            cavity elements. Otherwise, modify ring in-place
        rfmode              Selection mode: FUNDAMENTAL (default) keeps ratio
                                            between fundamental and harmonics,
                                            inputs are applied to the fundmental
                                            UNIQUE checks that only a single cavity
                                            is present
                                            ALL set all cavities
                            FUNDAMENTAL and UNIQUE require scalr inputs, ALL requires
                            vectors with shape (n_cavities,)
        method              ANALYTIC computes frequency form dp, ct and harmonic
                            number [default]
                            TRACKING cancels average orbit
        bpms                refpts used for average orbit calculation
        df                  delta used for frequency response
    """

    def av_horbit(ring, bpms):
        if bpms is None:
            bpms = range(len(ring))
        _, o = ring.find_orbit(refpts=bpms)
        return numpy.mean(o[:,0])

    def rf_resp(ring, df, bpms):
        frequency = ring.get_rf_frequency(cavpts=cavpts)
        ringp = ring.set_rf_frequency(frequency+0.5*df,
                                      cavpts=cavpts, copy=True)
        ringm = ring.set_rf_frequency(frequency-0.5*df,
                                      cavpts=cavpts, copy=True)
        op = av_horbit(ringp, bpms)
        om = av_horbit(ringm, bpms)
        return (op-om)/df

    if frequency is None:
        if method is FRFMethod.TRACKING and ring.radiation:
            frequency = ring.get_rf_frequency(cavpts=cavpts)
            resp_rf = rf_resp(ring, df, bpms)
            avorb = av_horbit(ring, bpms)
            frequency -= avorb/resp_rf
        elif method is FRFMethod.ANALYTIC or not ring.radiation:
            frequency = ring.get_revolution_frequency(dp=dp, dct=dct) \
                        * ring.harmonic_number
        else:
            raise AtError('Unknow FRFMethod in set_rf_frequency')
    return set_cavity(ring, Frequency=frequency, cavpts=cavpts, copy=copy,
                      rfmode=rfmode)


Lattice.mcf = property(get_mcf, doc="Momentum compaction factor")
Lattice.slip_factor = property(get_slip_factor, doc="Slip factor")
Lattice.get_revolution_frequency = get_revolution_frequency
Lattice.revolution_frequency = property(get_revolution_frequency,
                                        doc="Revolution frequency"
                                        " of on-momentum particles"
                                        " (full ring) [Hz]")
Lattice.get_mcf = get_mcf
Lattice.get_slip_factor = get_slip_factor
Lattice.set_rf_frequency = set_rf_frequency
Lattice.rf_frequency = property(get_rf_frequency, set_rf_frequency,
                                doc="Fundamental RF frequency [Hz]")
