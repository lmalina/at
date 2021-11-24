from enum import Enum
import numpy
from .elements import RFCavity
from .utils import AtError, checktype, make_copy, checkattr
from .lattice_object import Lattice

__all__ = ['get_rf_frequency', 'get_rf_voltage', 'set_rf_voltage',
           'get_rf_timelag', 'set_rf_timelag', 'set_cavity', 
           'set_rf_frequency', 'Frf', 'RFMode', 'get_rf_cavities']


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


def get_rf_voltage_array(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the array of RF voltages
    KEYWORDS
        cavpts=None   Cavity location. If None, look for ring.cavpts,
                      otherwise take all cavities. This allows to ignore
                      harmonic cavities.

        rfmode        Selection mode: FUNDAMENTAL (default) returns the minimum, 
                                      UNIQUE checks that only a single cavity 
                                      is present
          
                            ALL returns all cavities
    """
    vcell = _get_rf_attr(ring, 'Voltage', cavpts=cavpts, rfmode=rfmode)
    if rfmode is RFMode.UNIQUE or rfmode is RFMode.FUNDAMENTAL:
       n_cavity = len(get_rf_cavities(ring, cavpts=cavpts, rfmode=rfmode))
       print(n_cavity)
       vcell = numpy.broadcast_to(vcell,(n_cavity,))
    return vcell


def get_rf_voltage(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the total RF voltage
    KEYWORDS
        cavpts=None   Cavity location. If None, look for ring.cavpts,
                      otherwise take all cavities. This allows to ignore
                      harmonic cavities.

        rfmode        Selection mode: FUNDAMENTAL (default) returns the minimum, 
                                      UNIQUE checks that only a single cavity 
                                      is present
                                      ALL returns all cavities
    """
    vcell = get_rf_voltage_array(ring, cavpts=cavpts, rfmode=rfmode)
    return ring.periodicity * sum(vcell)


def get_rf_timelag(ring, cavpts=None, rfmode=RFMode.FUNDAMENTAL):
    """Return the RF time lag
    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
    """
    return _get_rf_attr(ring, 'TimeLag', cavpts=cavpts, rfmode=rfmode)


def set_rf_frequency(ring, frequency, cavpts=None, copy=False, rfmode=RFMode.FUNDAMENTAL):
    """Set the RF voltage

    PARAMETERS
        ring                lattice description
        frequency           RF frequency

    KEYWORDS
        cavpts=None         Cavity location. If None, look for ring.cavpts,
                            otherwise take all cavities. This allows to ignore
                            harmonic cavities.
        copy=False          If True, returns a shallow copy of ring with new
                            cavity elements. Otherwise, modify ring in-place
        rfmode              Selection mode: FUNDAMENTAL (default) keeps ratio between
                                            fundamental and harmonics, inputs is applied
                                            to the fundmental 
                                            UNIQUE checks that only a single cavity 
                                            is present
                                            ALL set all cavities
                            FUNDAMENTAL and UNIQUE require scalr inputs, ALL requires
                            vectors with shape (n_cavities,)
    """
    return set_cavity(ring, Frequency=frequency, cavpts=cavpts, copy=copy, rfmode=rfmode)


def set_rf_voltage(ring, voltage, cavpts=None, copy=False, rfmode=RFMode.FUNDAMENTAL):
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
        rfmode              Selection mode: FUNDAMENTAL (default) apply scale factor
                                            to all cavities 
                                            UNIQUE checks that only a single cavity 
                                            is present
                                            ALL set all cavities
                            FUNDAMENTAL and UNIQUE require scalr inputs, ALL requires
                            vectors with shape (n_cavities,)
    """
    return set_cavity(ring, Voltage=voltage, cavpts=cavpts, copy=copy, rfmode=rfmode)


def set_rf_timelag(ring, timelag, cavpts=None, copy=False, rfmode=RFMode.FUNDAMENTAL):
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
        rfmode              Selection mode: FUNDAMENTAL (default) apply Timelag to
                                            the fundmanetal cavities and keeps 
                                            difference w.r.t harmonic systems
                                            UNIQUE checks that only a single cavity 
                                            is present
                                            ALL set all cavities
                            FUNDAMENTAL and UNIQUE require scalar inputs, ALL requires
                            vectors with shape (n_cavities,)
    """
    return set_cavity(ring, TimeLag=timelag, cavpts=cavpts, copy=copy, rfmode=rfmode)


# noinspection PyPep8Naming
def set_cavity(ring, Voltage=None, Frequency=None, TimeLag=None, cavpts=None,
               copy=False, rfmode=RFMode.FUNDAMENTAL):
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
        rfmode              Selection mode: FUNDAMENTAL (default) maintains relations
                                            w.r.t harmonic systems
                                            UNIQUE checks that only a single cavity 
                                            is present
                                            ALL set all cavities
                            FUNDAMENTAL and UNIQUE require scalr inputs, ALL requires
                            vectors with shape (n_cavities,)
    """
    if cavpts is None:
        cavpts = getattr(ring, 'cavpts', checktype(RFCavity))
    cavities = get_rf_cavities(ring, cavpts=cavpts, rfmode=rfmode)
    n_cavities = len(cavities)
    if n_cavities < 1:
        raise AtError('No cavity found in the lattice')

    if rfmode is RFMode.UNIQUE or rfmode is RFMode.FUNDAMENTAL:
        cond = ((numpy.isscalar(Frequency) or Frequency is Frf.NOMINAL
                 or Frequency is None)
                and (numpy.isscalar(TimeLag) or TimeLag is None)
                and (numpy.isscalar(Voltage) or Voltage is None))
        if not cond:
            raise AtError('RFMode.UNIQUE and RFMode.FUNDAMENTAL require scalar inputs')

    modif = {}
    if Frequency is not None:
        if Frequency is Frf.NOMINAL:
            Frequency = ring.revolution_frequency * ring.harmonic_number
        if rfmode is RFMode.FUNDAMENTAL:
            fall = get_rf_frequency(ring, cavpts=cavpts, rfmode=RFMode.ALL) 
            ffund = get_rf_frequency(ring, cavpts=cavpts, rfmode=RFMode.FUNDAMENTAL)
            Frequency *= fall/ffund
        modif['Frequency'] = Frequency

    if TimeLag is not None:
        if rfmode is RFMode.FUNDAMENTAL:
            tall = get_rf_timelag(ring, cavpts=cavpts, rfmode=RFMode.ALL) 
            tfund = get_rf_timelag(ring, cavpts=cavpts, rfmode=RFMode.FUNDAMENTAL)
            TimeLag += tall-tfund
        modif['TimeLag'] = TimeLag
        
    if Voltage is not None:
        if  rfmode is RFMode.FUNDAMENTAL:
            vall = get_rf_voltage_array(ring, cavpts=cavpts, rfmode=RFMode.ALL) 
            Voltage *= vall/sum(vall) * n_cavities
        modif['Voltage'] = Voltage  / ring.periodicity / n_cavities

    # noinspection PyShadowingNames
    @make_copy(copy)
    def apply(ring, cavpts, modif):
        for attr in modif.keys():
            values = modif[attr]
            try:
                values = numpy.broadcast_to(values, (ring.refcount(cavpts),))          
            except ValueError:
                raise AtError('set_cavity args should be either scalar or a vector (ncavs,)')
            for val, cavity in zip(values, ring.select(cavpts)):
                cavity.update({attr:val})

    return apply(ring, cavpts, modif)


Lattice.get_rf_voltage = get_rf_voltage
Lattice.get_rf_frequency = get_rf_frequency
Lattice.get_rf_timelag = get_rf_timelag
Lattice.set_rf_voltage = set_rf_voltage
Lattice.set_rf_frequency = set_rf_frequency
Lattice.set_rf_timelag = set_rf_timelag
Lattice.get_rf_cavities = get_rf_cavities
Lattice.set_cavity = set_cavity
