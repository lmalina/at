"""
Load lattice from Matlab file.

This is working from a specific file and may not be general.
"""
import scipy.io
from .lattice import elements
import numpy

CLASS_MAPPING = {'Quad': 'Quadrupole', 'Sext': 'Sextupole', 'Bend': 'Dipole', 'AP': 'Aperture',
                 'RF': 'RFCavity', 'BPM': 'Monitor'}

CLASSES = set(['Marker', 'Monitor', 'Aperture', 'Drift', 'ThinMultipole', 'Multipole', 'Dipole', 'Quadrupole',
               'Sextupole', 'Octupole', 'RFCavity', 'RingParam', 'M66', 'Corrector'])

FAMILY_MAPPING = {'ap': 'Aperture', 'rf': 'RFCavity', 'bpm': 'Monitor'}

PASSMETHOD_MAPPING = {'CorrectorPass': 'Corrector', 'Matrix66Pass': 'M66',
                      'CavityPass': 'RFCavity', 'RFCavityPass': 'RFCavity',
                      'QuadLinearPass': 'Quadrupole', 'BendLinearPass': 'Dipole',
                      'BndMPoleSymplectic4Pass': 'Dipole', 'BndMPoleSymplectic4RadPass': 'Dipole',
                      'AperturePass': 'Aperture', 'ThinMPolePass': 'ThinMultipole',
                      'DriftPass': 'Drift'}


def hasattrs(kwargs, *attributes):
    """Check if the element would have the specified attribute(s), i.e. if they
        are in kwargs; allows checking for multiple attributes in one go.

    Args:
        kwargs (dict): The dictionary of keyword arguments passed to the
                        Element constructor.
        attributes (iterable): A list of strings, the attribute names to be
                                checked.

    Returns:
        bool: A single boolean, True if the element has any of the specified
               attributes.
    """
    for attribute in attributes:
        if attribute in kwargs:
            return True
    return False


def find_class_name(kwargs):
    """Attempts to correctly identify the Class of the element from its kwargs.

    Args:
        kwargs (dict): The dictionary of keyword arguments passed to the
                        Element constructor.

    Returns:
        str: The guessed Class name, as a string.

    Raises:
        AttributeError: if the Class name found in kwargs is not a valid Class
                         in pyAT.
    """

    def low_order(key):
        polynom = numpy.array(kwargs[key], dtype=numpy.float64).reshape(-1)
        try:
            low = numpy.where(polynom != 0.0)[0][0]
        except IndexError:
            low = -1
        return low

    try:
        class_name = kwargs.pop('Class')
        class_name = CLASS_MAPPING.get(class_name, class_name)
        if class_name in CLASSES:
            return class_name
        else:
            raise AttributeError("Class {0} does not exist.".format(class_name))
    except KeyError:
        fam_name = kwargs.get('FamName')
        if fam_name in CLASSES:
            return fam_name
        elif fam_name.lower() in FAMILY_MAPPING.keys():
            return FAMILY_MAPPING[fam_name.lower()]
        else:
            pass_method = kwargs.get('PassMethod')
            class_from_pass = PASSMETHOD_MAPPING.get(pass_method)
            if class_from_pass is not None:
                return class_from_pass
            else:
                length = float(kwargs.get('Length', 0.0))
                if hasattrs(kwargs, 'FullGap', 'FringeInt1', 'FringeInt2', 'gK', 'EntranceAngle', 'ExitAngle'):
                    return 'Dipole'
                elif hasattrs(kwargs, 'Voltage', 'Frequency', 'HarmNumber', 'PhaseLag', 'TimeLag'):
                    return 'RFCavity'
                elif hasattrs(kwargs, 'Periodicity'):
                    return 'RingParam'
                elif hasattrs(kwargs, 'Limits'):
                    return 'Aperture'
                elif hasattrs(kwargs, 'M66'):
                    return 'M66'
                elif hasattrs(kwargs, 'K'):
                    return 'Quadrupole'
                elif hasattrs(kwargs, 'PolynomB'):
                    loworder = low_order('PolynomB')
                    if loworder == 1:
                        return 'Quadrupole'
                    elif loworder == 2:
                        return 'Sextupole'
                    elif loworder == 3:
                        return 'Octupole'
                    elif (pass_method in {'StrMPoleSymplectic4Pass', 'StrMPoleSymplectic4RadPass'}) or (length > 0):
                        return 'Multipole'
                    else:
                        return 'ThinMultipole'
                elif hasattrs(kwargs, 'KickAngle'):
                    return 'Corrector'
                elif length > 0.0:
                    return 'Drift'
                elif hasattrs(kwargs, 'GCR'):
                    return 'Monitor'
                else:
                    return 'Marker'


def element_from_dict(elem_dict, index=None, check=True):
    """return an AT element from a dictinary of attributes
    """

    def sanitise_class(index, class_name, kwargs):
        """Checks that the Class and PassMethod of the element are a valid
            combination. Some Classes and PassMethods are incompatible and would
            raise errors during calculation if left, so we raise an error here with
            a more helpful message.

        Args:
            index:          element index
            class_name:     Proposed class name
            kwargs (dict):  The dictionary of keyword arguments passed to the
                            Element constructor.

        Raises:
            AttributeError: if the PassMethod and Class are incompatible.
        """

        def error_message(message, *args):
            location = '' if index is None else 'Error in element {0}: '.format(index)
            return location + 'PassMethod {0} is not compatible with '.format(pass_method) + message.format(*args)

        pass_method = kwargs.get('PassMethod')
        if pass_method is not None:
            pass_to_class = PASSMETHOD_MAPPING.get(pass_method)
            length = float(kwargs.get('Length', 0.0))
            if (pass_method == 'IdentityPass') and (length != 0.0):
                raise AttributeError(error_message("length {0}.", length))
            elif pass_to_class is not None:
                if pass_to_class != class_name:
                    raise AttributeError(error_message("Class {0}.", class_name))
            elif class_name in ['Marker', 'Monitor', 'RingParam']:
                if pass_method != 'IdentityPass':
                    raise AttributeError(error_message("Class {0}.", class_name))
            elif class_name == 'Drift':
                if pass_method != 'DriftPass':
                    raise AttributeError(error_message("Class {0}.", class_name))

    class_name = find_class_name(elem_dict)
    if check:
        sanitise_class(index, class_name, elem_dict)
    cl = getattr(elements, class_name)
    # Remove mandatory attributes from the keyword arguments.
    elem_args = [elem_dict.pop(attr) for attr in cl.REQUIRED_ATTRIBUTES]
    element = cl(*elem_args, **elem_dict)
    return element


def load_element(index, element_array, check=True):
    """Load what scipy produces into a pyat element object.
    """
    kwargs = {}
    for field_name in element_array.dtype.fields:
        # Remove any surplus dimensions in arrays.
        data = numpy.squeeze(element_array[field_name])
        # Convert strings in arrays back to strings.
        if data.dtype.type is numpy.unicode_:
            data = str(data)
        kwargs[field_name] = data

    return element_from_dict(kwargs, index=index, check=check)


def load(filename, key='RING', check=True):
    """Load a matlab at structure into a Python at list
    """
    m = scipy.io.loadmat(filename)
    element_arrays = m[key].flat
    return [load_element(i, elem[0][0], check=check) for (i, elem) in enumerate(element_arrays)]
