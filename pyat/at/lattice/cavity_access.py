from enum import Enum
import numpy
from .elements import RFCavity
from .utils import AtError, checktype, make_copy, checkattr
from .lattice_object import Lattice

__all__ = ['get_rf_frequency', 'get_rf_voltage', 'set_rf_voltage',
           'get_rf_timelag', 'set_rf_timelag', 'set_cavity', 'Frf']


class Frf(Enum):
    """Enum class for frequency setting"""
    NOMINAL = 'nominal'


class RFMode(Enum):
    UNIQUE = 'unique'
    FUNDAMENTAL = 'fundamental'
    ALL = 'all'


def _get_rf_attr(ring, attr, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    if cavpts is None:
        cavpts = getattr(ring, 'cavpts', checktype(RFCavity))

    all_attr = numpy.array([getattr(cavity, attr) for cavity in ring.select(cavpts)])
    if len(all_attr) == 0:
        raise AtError('No cavity found in the lattice')

    if rfmode is RFMode.ALL:
        return all_attr
    elif rfmode is RFMode.UNIQUE:
        rf_att = numpy.unique(all_attr)
        if len(rf_att) > 1:
            raise AtError('{0} not equal for all cavities'.format(attr))
        else:
            return rf_att[0]
    elif rfmode is RFMode.FUNDAMENTAL:
        freqmin = min(numpy.array([getattr(cavity, 'Frequency') for cavity
                                   in  ring.select(cavpts)]))   
        fund_cavs = filter(checkattr('Frequency', freqmin), ring)     
        rf_att = numpy.unique([getattr(cavity, attr) for cavity in fund_cavs])
        if len(rf_att) > 1:
            raise AtError('{0} not equal for all fundamental cavities'.format(attr))
        else:
            return rf_att[0]
    else:
        raise AtError('Unknown RFMode {0}'.format(rfmode))

       
def get_rf_cavities(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the cavity elements with fundamental RF frequency
    KEYWORDS
        cavpts=None   Cavity location. If None, look for ring.cavpts,
                      otherwise take all cavities. This allows to ignore
                      harmonic cavities.

        rfmode        Selection mode: FUNDAMENTAL (default) returns the minimum, 
                                      UNIQUE checks that only a single cavity 
                                      is present
                                      ALL returns all cavities
    """
    if cavpts is None:
        cavpts = getattr(ring, 'cavpts', checktype(RFCavity))
    if rfmode is RFMode.ALL:
        return list(ring.select(cavpts))   
    else:
        filtfunc = checkattr('Frequency', get_rf_frequency(ring, cavpts=cavpts, rfmode=rfmode))
        return list(filter(filtfunc, ring))
    

def get_rf_frequency(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the RF frequency
    KEYWORDS
        cavpts=None   Cavity location. If None, look for ring.cavpts,
                      otherwise take all cavities. This allows to ignore
                      harmonic cavities.

        rfmode        Selection mode: FUNDAMENTAL (default) returns the minimum, 
                                      UNIQUE checks that only a single cavity 
                                      is present
                                      ALL returns all cavities
                            
    """
    return _get_rf_attr(ring, 'Frequency', cavpts=cavpts, rfmode=rfmode)


def get_rf_voltage(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the total RF voltage for the full ring
    KEYWORDS
        cavpts=None   Cavity location. If None, look for ring.cavpts,
                      otherwise take all cavities. This allows to ignore
                      harmonic cavities.

        rfmode        Selection mode: FUNDAMENTAL (default) returns the minimum, 
                                      UNIQUE checks that only a single cavity 
                                      is present
                                      ALL returns all cavities
    """
    vcell = sum(_get_rf_attr(ring, 'Voltage', cavpts=cavpts, rfmode=rfmode))
    return ring.periodicity * vcell


def get_rf_timelag(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the RF time lag
    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
    """
    return _get_rf_attr(ring, 'TimeLag', cavpts=cavpts, rfmode=rfmode)


def set_rf_voltage(ring, voltage, cavpts=None, copy=False):
    """Set the RF voltage

    PARAMETERS
        ring                lattice description
        voltage             Total RF voltage for the full ring

    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
        copy=False          If True, returns a shallow copy of ring with new
                            cavity elements. Otherwise, modify ring in-place
    """
    return set_cavity(ring, Voltage=voltage, cavpts=cavpts, copy=copy)


def set_rf_timelag(ring, timelag, cavpts=None, copy=False):
    """Set the RF time lag

    PARAMETERS
        ring                lattice description
        timelag

    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
        copy=False          If True, returns a shallow copy of ring with new
                            cavity elements. Otherwise, modify ring in-place
    """
    return set_cavity(ring, TimeLag=timelag, cavpts=cavpts, copy=copy)


# noinspection PyPep8Naming
def set_cavity(ring, Voltage=None, Frequency=None, TimeLag=None, cavpts=None,
               copy=False):
    """
    Set the parameters of the RF cavities

    PARAMETERS
        ring                lattice description

    KEYWORDS
        Frequency=None      RF frequency. The special enum value Frf.NOMINAL
                            sets the frequency to the nominal value, given
                            ring length and harmonic number.
        Voltage=None        RF voltage for the full ring.
        TimeLag=None
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
        copy=False          If True, returns a shallow copy of ring with new
                            cavity elements. Otherwise, modify ring in-place
    """
    if cavpts is None:
        cavpts = getattr(ring, 'cavpts', checktype(RFCavity))
    n_cavities = ring.refcount(cavpts)
    if n_cavities < 1:
        raise AtError('No cavity found in the lattice')

    modif = {}
    if Frequency is not None:
        if Frequency is Frf.NOMINAL:
            Frequency = ring.revolution_frequency * ring.harmonic_number 
        modif['Frequency'] = Frequency
    if TimeLag is not None:
        modif['TimeLag'] = TimeLag
    if Voltage is not None:
        modif['Voltage'] = Voltage / ring.periodicity / n_cavities

    # noinspection PyShadowingNames
    @make_copy(copy)
    def apply(ring, cavpts, modif):
        for cavity in ring.select(cavpts):
            cavity.update(modif)

    return apply(ring, cavpts, modif)


Lattice.get_rf_voltage = get_rf_voltage
Lattice.get_rf_frequency = get_rf_frequency
Lattice.get_rf_timelag = get_rf_timelag
Lattice.set_rf_voltage = set_rf_voltage
Lattice.set_rf_timelag = set_rf_timelag
Lattice.set_cavity = set_cavity
