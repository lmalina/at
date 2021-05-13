"""
Coupled or non-coupled 4x4 linear motion
"""
import numpy
from math import sqrt, atan2, pi
from at.lattice import Lattice, check_radiation, get_s_pos, \
    bool_refpts, DConstant
from at.tracking import lattice_pass
from at.physics import find_orbit4, find_m44, jmat, find_orbit
from .harmonic_analysis import get_tunes_harmonic

__all__ = ['linopt', 'linopt2', 'linopt4', 'avlinopt', 'get_mcf', 'get_tune',
           'get_chrom']

DDP = 1e-8

_jmt = jmat(1)

# dtype for structured array containing linopt parameters
_DATA1_DTYPE = [('alpha', numpy.float64, (2,)),
                ('beta', numpy.float64, (2,)),
                ('mu', numpy.float64, (2,)),
                ('gamma', numpy.float64),
                ('A', numpy.float64, (2, 2)),
                ('B', numpy.float64, (2, 2)),
                ('C', numpy.float64, (2, 2)),
                ('s_pos', numpy.float64),
                ('closed_orbit', numpy.float64, (6,)),
                ('dispersion', numpy.float64, (4,)),
                ('m44', numpy.float64, (4, 4))]
#('idx', numpy.uint32),
#('W', numpy.float64, (2,))
_DATA2_DTYPE = [('alpha', numpy.float64, (2,)),
                ('beta', numpy.float64, (2,)),
                ('mu', numpy.float64, (2,)),
                ('s_pos', numpy.float64),
                ('closed_orbit', numpy.float64, (6,)),
                ('dispersion', numpy.float64, (4,)),
                ('M', numpy.float64, (4, 4))]

_DATA4_DTYPE = [('alpha', numpy.float64, (2,)),
                ('beta', numpy.float64, (2,)),
                ('mu', numpy.float64, (2,)),
                ('gamma', numpy.float64),
                ('s_pos', numpy.float64),
                ('closed_orbit', numpy.float64, (6,)),
                ('dispersion', numpy.float64, (4,)),
                ('M', numpy.float64, (4, 4))]

_W_DTYPE = [('W', numpy.float64, (2,))]


def _twiss22(ms, alpha0, beta0):
    """
    Calculate Twiss parameters from the standard 2x2 transfer matrix
    (i.e. x or y).
    """
    bbb = ms[0, 1]
    aaa = ms[0, 0] * beta0 - bbb * alpha0
    beta = (aaa * aaa + bbb * bbb) / beta0
    alpha = -(aaa * (ms[1, 0] * beta0 - ms[1, 1] * alpha0) +
              bbb * ms[1, 1]) / beta0
    mu = numpy.arctan2(bbb, aaa)
    # # Unwrap negative jumps in betatron phase advance
    # dmu = numpy.diff(numpy.append([0], mu))
    # jumps = dmu < 0
    # mu += numpy.cumsum(jumps) * 2.0 * numpy.pi
    return alpha, beta, mu


def _closure(m22):
    diff = (m22[0, 0] - m22[1, 1]) / 2.0
    sinmu = numpy.sign(m22[0, 1]) * sqrt(-m22[0, 1] * m22[1, 0] - diff * diff)
    cosmu = 0.5 * numpy.trace(m22)
    alpha = diff / sinmu
    beta = m22[0, 1] / sinmu
    tune = (atan2(sinmu, cosmu) / 2.0 / pi) % 1
    return alpha, beta, tune


def _unwrap(mu):
    """Remove the phase jumps"""
    dmu = numpy.diff(numpy.concatenate((numpy.zeros((1, 2)), mu)), axis=0)
    jumps = dmu < -1.e-3
    mu += numpy.cumsum(jumps, axis=0) * 2.0 * numpy.pi


# noinspection PyShadowingNames,PyPep8Naming
def _linopt(ring, analyze, dp=0.0, refpts=None, orbit=None, twiss_in=None,
            keep_lattice=False, get_chrom=False, get_w=False,
            **kwargs):
    """"""
    def scan(ring, dp, refpts, orb0):
        m44, mstack = find_m44(ring, dp, refpts, orbit=orb0,
                               keep_lattice=True, **kwargs)
        tune, el0, els = analyze(m44, mstack)
        return tune, el0, els, m44, iter(mstack)

    def wget(ddp, elup, eldn):
        alpha_up, beta_up = elup[:2]
        alpha_dn, beta_dn = eldn[:2]
        db0 = (beta_up - beta_dn) / ddp
        mb0 = (beta_up + beta_dn) / 2
        da0 = (alpha_up - alpha_dn) / ddp
        ma0 = (alpha_up + alpha_dn) / 2
        w0 = numpy.sqrt((da0 - ma0 / mb0 * db0) ** 2 + (db0 / mb0) ** 2)
        return w0

    dp_step = kwargs.get('DPStep', DConstant.DPStep)

    # Get initial orbit
    dpup = dp + 0.5*dp_step
    dpdn = dp - 0.5*dp_step
    if twiss_in is None:
        orb0, orbs = find_orbit(ring, refpts, dp=dp, orbit=orbit, **kwargs)
        o0up, oup = find_orbit4(ring, dp=dpup, refpts=refpts,
                                keep_lattice=keep_lattice)
        o0dn, odn = find_orbit4(ring, dp=dpdn, refpts=refpts,
                                keep_lattice=True)
    else:
        orb0 = numpy.zeros((6,)) if orbit is None else orbit
        try:
            d0 = twiss_in['dispersion']
        except KeyError:
            print('Dispersion not found in twiss_in, setting to zero')
            d0 = numpy.zeros((4,))
        dorbit = numpy.hstack((0.5 * dp_step * d0,
                               numpy.array([0.5 * dp_step, 0])))

        orb0, orbs = find_orbit(ring, refpts, dp=dp, orbit=orb0,
                                keep_lattice=keep_lattice, **kwargs)
        o0up, oup = find_orbit4(ring, dp=dpup, refpts=refpts, orbit=orb0+dorbit,
                                keep_lattice=keep_lattice)
        o0dn, odn = find_orbit4(ring, dp=dpdn, refpts=refpts, orbit=orb0-dorbit,
                                keep_lattice=True)

    d0 = (o0up - o0dn)[:4] / dp_step
    ds = ((up - dn)[:4] / dp_step for up, dn in zip(oup, odn))
    tune, el0, els, m44, ms = scan(ring, dp, refpts, orb0)
    spos = get_s_pos(ring, refpts)

    data0 = [*el0, get_s_pos(ring, len(ring)), orb0, d0, m44]
    datas = [els, iter(spos), iter(orbs), ds, ms]

    if get_w:
        tuneup, el0up, elsup, _, _ = scan(ring, dpup, refpts, o0up)
        tunedn, el0dn, elsdn, _, _ = scan(ring, dpdn, refpts, o0dn)
        chrom = (tuneup - tunedn) / dp_step
        data0.append(wget(dp_step, el0up, el0dn))
        datas.append((wget(dp_step, elup, eldn)
                      for elup, eldn in zip(elsup, elsdn)))
    elif get_chrom:
        tuneup, el0up, elsup, _, _ = scan(ring, dpup, None, o0up)
        tunedn, el0dn, elsdn, _, _ = scan(ring, dpdn, None, o0dn)
        chrom = (tuneup - tunedn) / dp_step
    else:
        chrom = numpy.NaN

    beamdata = numpy.array((tune, chrom),
                           dtype=[('tune', numpy.float64, (2,)),
                                  ('chromaticity', numpy.float64, (2,)),
                                  ]).view(numpy.recarray)

    return beamdata, data0, ((*el, *args) for el, *args in zip(*datas))


@check_radiation(False)
def linopt4(ring, refpts=None, dp=0.0, get_w=False, **kwargs):
    """Perform linear analysis of a lattice
    elemdata0, beamdata, elemdata = linopt4(ring, refpts, **kwargs)
    PARAMETERS
        ring            lattice description.
        refpts=None     elements at which data is returned. It can be:
                        1) an integer in the range [-len(ring), len(ring)-1]
                           selecting the element according to python indexing
                           rules. As a special case, len(ring) is allowed and
                           refers to the end of the last element,
                        2) an ordered list of such integers without duplicates,
                        3) a numpy array of booleans of maximum length
                           len(ring)+1, where selected elements are True.
    KEYWORDS
        dp=0.0          momentum deviation.
        orbit           avoids looking for the closed orbit if is already known
                        ((6,) array)
        get_chrom=False compute dispersion and chromaticities. Needs computing
                        the tune and orbit at 2 different momentum deviations
                        around the central one.
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        XYStep=1.0e-8   transverse step for numerical computation
        DPStep=1.0E-6   momentum deviation used for computation of
                        chromaticities and dispersion
        coupled=True    if False, simplify the calculations by assuming
                        no H/V coupling
        twiss_in=None   Initial twiss to compute transfer line optics of the
                        type lindata, the initial orbit in twiss_in is ignored,
                        only the beta and alpha are required other quatities
                        set to 0 if absent
        get_w=False     computes chromatic amplitude functions (W) [4], need to
                        compute the optics at 2 different momentum deviations
                        around the central one.
    OUTPUT
        lindata0        linear optics data at the entrance/end of the ring
        tune            [tune_A, tune_B], linear tunes for the two normal modes
                        of linear motion [1]
        chrom           [ksi_A , ksi_B], chromaticities ksi = d(nu)/(dP/P).
                        Only computed if 'get_chrom' is True
        lindata         linear optics at the points refered to by refpts, if
                        refpts is None an empty lindata structure is returned.

        lindata is a record array with fields:
        idx             element index in the ring
        s_pos           longitudinal position [m]
        closed_orbit    (6,) closed orbit vector
        dispersion      (4,) dispersion vector
        W               (2,) chromatic amplitude function
                        Only computed if 'get_chrom' is True
        M               (4, 4) transfer matrix M from the beginning of ring
                        to the entrance of the element [2]
        mu              [mux, muy], betatron phase (modulo 2*pi)
        beta            [betax, betay] vector
        alpha           [alphax, alphay] vector
        A               (2, 2) matrix A in [3]
        B               (2, 2) matrix B in [3]
        C               (2, 2) matrix C in [3]
        gamma           gamma parameter of the transformation to eigenmodes
        All values given at the entrance of each element specified in refpts.
        Field values can be obtained with either
        lindata['idx']    or
        lindata.idx
    REFERENCES
        [1] D.Edwars,L.Teng IEEE Trans.Nucl.Sci. NS-20, No.3, p.885-888, 1973
        [2] E.Courant, H.Snyder
        [3] D.Sagan, D.Rubin Phys.Rev.Spec.Top.-Accelerators and beams,
            vol.2 (1999)
        [4] Brian W. Montague Report LEP Note 165, CERN, 1979
    """
    def _analyze4(m44, mstack):

        def grp1(mi):
            mm = mi[:2, :2]
            nn = mi[2:, 2:]
            m = mi[:2, 2:]
            n = mi[2:, :2]
            gamma = sqrt(numpy.linalg.det(numpy.dot(n, C) + g*nn))
            e12 = (g * mm - m.dot(_jmt.dot(C.T.dot(_jmt.T)))) / gamma
            f12 = (n.dot(C) + g * nn) / gamma
            alpha_a, beta_a, mu_a = _twiss22(e12, alpha0_a, beta0_a)
            alpha_b, beta_b, mu_b = _twiss22(f12, alpha0_b, beta0_b)
            return numpy.array([alpha_a, alpha_b]),\
                   numpy.array([beta_a, beta_b]),\
                   numpy.array([mu_a, mu_b]), gamma

        M = m44[:2, :2]
        N = m44[2:, 2:]
        m = m44[:2, 2:]
        n = m44[2:, :2]
        H = m + _jmt.dot(n.T.dot(_jmt.T))
        t = numpy.trace(M - N)
        t2 = t * t
        t2h = t2 + 4.0 * numpy.linalg.det(H)
        g2 = (1.0 + sqrt(t2 / t2h)) / 2
        g = sqrt(g2)
        C = -H * numpy.sign(t) / (g * sqrt(t2h))
        A = g2*M - g*(m.dot(_jmt.dot(C.T.dot(_jmt.T))) + C.dot(n)) + \
            C.dot(N.dot(_jmt.dot(C.T.dot(_jmt.T))))
        B = g2*N + g*(_jmt.dot(C.T.dot(_jmt.T.dot(m))) + n.dot(C)) + \
            _jmt.dot(C.T.dot(_jmt.T.dot(M.dot(C))))
        alpha0_a, beta0_a, tune_a = _closure(A)
        alpha0_b, beta0_b, tune_b = _closure(B)
        inival = (numpy.array([alpha0_a, alpha0_b]),
                  numpy.array([beta0_a, beta0_b]),
                  numpy.zeros((2,)), g)
#       inival = grp1(numpy.identity(6))
        rv = (grp1(ms44) for ms44 in mstack)
        return numpy.array([tune_a, tune_b]), inival, rv

    beamdata, data0, datas = _linopt(ring, refpts=refpts, dp=dp,
                                     analyze=_analyze4, get_w=get_w,
                                     **kwargs)
    dtype = _DATA4_DTYPE + _W_DTYPE if get_w else _DATA4_DTYPE
    elemdata0 = numpy.array(tuple(data0), dtype=dtype).view(numpy.recarray)
    elemdata = numpy.fromiter(datas, dtype, count=ring.refcount(refpts)
                              ).view(numpy.recarray)
    _unwrap(elemdata.mu)
    return elemdata0, beamdata, elemdata


@check_radiation(False)
def linopt2(ring, refpts=None, dp=0.0, get_w=False, **kwargs):
    """
    Perform linear analysis of a lattice
    elemdata0, beamdata, elemdata = linopt2(ring, refpts, **kwargs)
    PARAMETERS
        ring            lattice description.
        refpts=None     elements at which data is returned. It can be:
                        1) an integer in the range [-len(ring), len(ring)-1]
                           selecting the element according to python indexing
                           rules. As a special case, len(ring) is allowed and
                           refers to the end of the last element,
                        2) an ordered list of such integers without duplicates,
                        3) a numpy array of booleans of maximum length
                           len(ring)+1, where selected elements are True.
    KEYWORDS
        dp=0.0          momentum deviation.
        orbit           avoids looking for the closed orbit if is already known
                        ((6,) array)
        get_chrom=False compute dispersion and chromaticities. Needs computing
                        the tune and orbit at 2 different momentum deviations
                        around the central one.
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        XYStep=1.0e-8   transverse step for numerical computation
        DPStep=1.0E-6   momentum deviation used for computation of
                        chromaticities and dispersion
        twiss_in=None   Initial twiss to compute transfer line optics of the
                        type lindata, the initial orbit in twiss_in is ignored,
                        only the beta and alpha are required other quatities
                        set to 0 if absent
        get_w=False     computes chromatic amplitude functions (W) [4], need to
                        compute the optics at 2 different momentum deviations
                        around the central one.
    OUTPUT
        lindata0        linear optics data at the entrance/end of the ring
        tune            [tune_A, tune_B], linear tunes for the two normal modes
                        of linear motion [1]
        chrom           [ksi_A , ksi_B], chromaticities ksi = d(nu)/(dP/P).
                        Only computed if 'get_chrom' is True
        lindata         linear optics at the points refered to by refpts, if
                        refpts is None an empty lindata structure is returned.

        lindata is a record array with fields:
        idx             element index in the ring
        s_pos           longitudinal position [m]
        closed_orbit    (6,) closed orbit vector
        dispersion      (4,) dispersion vector
        W               (2,) chromatic amplitude function
                        Only computed if 'get_chrom' is True
        M               (4, 4) transfer matrix M from the beginning of ring
                        to the entrance of the element [2]
        mu              [mux, muy], betatron phase (modulo 2*pi)
        beta            [betax, betay] vector
        alpha           [alphax, alphay] vector
        All values given at the entrance of each element specified in refpts.
        Field values can be obtained with either
        lindata['idx']    or
        lindata.idx
    REFERENCES
        [1] D.Edwards,L.Teng IEEE Trans.Nucl.Sci. NS-20, No.3, p.885-888, 1973
        [2] E.Courant, H.Snyder
        [3] D.Sagan, D.Rubin Phys.Rev.Spec.Top.-Accelerators and beams,
            vol.2 (1999)
        [4] Brian W. Montague Report LEP Note 165, CERN, 1979
    """
    def _analyze2(m44, mstack):

        def grp2(mi):
            ma = mi[:2, :2]
            mb = mi[2:, 2:]
            alpha_a, beta_a, mu_a = _twiss22(ma, alpha0_a, beta0_a)
            alpha_b, beta_b, mu_b = _twiss22(mb, alpha0_b, beta0_b)
            return numpy.array([alpha_a, alpha_b]),\
                   numpy.array([beta_a, beta_b]),\
                   numpy.array([mu_a, mu_b])

        A = m44[:2, :2]
        B = m44[2:, 2:]
        alpha0_a, beta0_a, tune_a = _closure(A)
        alpha0_b, beta0_b, tune_b = _closure(B)
        inival = (numpy.array([alpha0_a, alpha0_b]),
                  numpy.array([beta0_a, beta0_b]),
                  numpy.zeros((2,)))
        rv = (grp2(ms44) for ms44 in mstack)
        return numpy.array([tune_a, tune_b]), inival, rv

    beamdata, data0, datas = _linopt(ring, refpts=refpts, dp=dp,
                                     analyze=_analyze2, get_w=get_w,
                                     **kwargs)
    dtype = _DATA2_DTYPE + _W_DTYPE if get_w else _DATA2_DTYPE
    elemdata0 = numpy.array(tuple(data0), dtype=dtype).view(numpy.recarray)
    elemdata = numpy.fromiter(datas, dtype, count=ring.refcount(refpts)
                              ).view(numpy.recarray)
    _unwrap(elemdata.mu)
    return elemdata0, beamdata, elemdata


# noinspection PyPep8Naming
@check_radiation(False)
def linopt(ring, dp=0.0, refpts=None, get_chrom=False, coupled=True, get_w=False, **kwargs):
    """
    Perform linear analysis of a lattice
    lindata0, tune, chrom, lindata = linopt(ring, dp[, refpts])
    PARAMETERS
        ring            lattice description.
        dp=0.0          momentum deviation.
        refpts=None     elements at which data is returned. It can be:
                        1) an integer in the range [-len(ring), len(ring)-1]
                           selecting the element according to python indexing
                           rules. As a special case, len(ring) is allowed and
                           refers to the end of the last element,
                        2) an ordered list of such integers without duplicates,
                        3) a numpy array of booleans of maximum length
                           len(ring)+1, where selected elements are True.
    KEYWORDS
        orbit           avoids looking for the closed orbit if is already known
                        ((6,) array)
        get_chrom=False compute dispersion and chromaticities. Needs computing
                        the tune and orbit at 2 different momentum deviations
                        around the central one.
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        XYStep=1.0e-8   transverse step for numerical computation
        DPStep=1.0E-6   momentum deviation used for computation of
                        chromaticities and dispersion
        coupled=True    if False, simplify the calculations by assuming
                        no H/V coupling
        twiss_in=None   Initial twiss to compute transfer line optics of the
                        type lindata, the initial orbit in twiss_in is ignored,
                        only the beta and alpha are required other quatities
                        set to 0 if absent
        get_w=False     computes chromatic amplitude functions (W) [4], need to
                        compute the optics at 2 different momentum deviations
                        around the central one.
    OUTPUT
        lindata0        linear optics data at the entrance/end of the ring
        tune            [tune_A, tune_B], linear tunes for the two normal modes
                        of linear motion [1]
        chrom           [ksi_A , ksi_B], chromaticities ksi = d(nu)/(dP/P).
                        Only computed if 'get_chrom' is True
        lindata         linear optics at the points refered to by refpts, if
                        refpts is None an empty lindata structure is returned.

        lindata is a record array with fields:
        idx             element index in the ring
        s_pos           longitudinal position [m]
        closed_orbit    (6,) closed orbit vector
        dispersion      (4,) dispersion vector
        W               (2,) chromatic amplitude function
                        Only computed if 'get_chrom' is True
        m44             (4, 4) transfer matrix M from the beginning of ring
                        to the entrance of the element [2]
        mu              [mux, muy], betatron phase (modulo 2*pi)
        beta            [betax, betay] vector
        alpha           [alphax, alphay] vector
        All values given at the entrance of each element specified in refpts.
        In case coupled = True additional outputs are available:
        A               (2, 2) matrix A in [3]
        B               (2, 2) matrix B in [3]
        C               (2, 2) matrix C in [3]
        gamma           gamma parameter of the transformation to eigenmodes
        Field values can be obtained with either
        lindata['idx']    or
        lindata.idx
    REFERENCES
        [1] D.Edwars,L.Teng IEEE Trans.Nucl.Sci. NS-20, No.3, p.885-888, 1973
        [2] E.Courant, H.Snyder
        [3] D.Sagan, D.Rubin Phys.Rev.Spec.Top.-Accelerators and beams,
            vol.2 (1999)
        [4] Brian W. Montague Report LEP Note 165, CERN, 1979
    """
    def _analyze2(m44, mstack):

        def grp2(mi):
            ma = mi[:2, :2]
            mb = mi[2:, 2:]
            alpha_a, beta_a, mu_a = _twiss22(ma, alpha0_a, beta0_a)
            alpha_b, beta_b, mu_b = _twiss22(mb, alpha0_b, beta0_b)
            return numpy.array([alpha_a, alpha_b]),\
                   numpy.array([beta_a, beta_b]),\
                   numpy.array([mu_a, mu_b]),\
                   numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN

        A = m44[:2, :2]
        B = m44[2:, 2:]
        alpha0_a, beta0_a, tune_a = _closure(A)
        alpha0_b, beta0_b, tune_b = _closure(B)
        inival = (numpy.array([alpha0_a, alpha0_b]),
                  numpy.array([beta0_a, beta0_b]),
                  numpy.zeros((2,)), numpy.NaN, numpy.NaN, numpy.NaN, numpy.NaN)
        rv = (grp2(ms44) for ms44 in mstack)
        return numpy.array([tune_a, tune_b]), inival, rv

    def _analyze4(m44, mstack):
        def grp1(mi):
            mm = mi[:2, :2]
            nn = mi[2:, 2:]
            m = mi[:2, 2:]
            n = mi[2:, :2]
            gamma = sqrt(numpy.linalg.det(numpy.dot(n, C) + g*nn))
            e12 = (g * mm - m.dot(_jmt.dot(C.T.dot(_jmt.T)))) / gamma
            f12 = (n.dot(C) + g * nn) / gamma
            alpha_a, beta_a, mu_a = _twiss22(e12, alpha0_a, beta0_a)
            alpha_b, beta_b, mu_b = _twiss22(f12, alpha0_b, beta0_b)
            AL = e12.dot(A.dot(_jmt.dot(e12.T.dot(_jmt.T))))
            BL = f12.dot(B.dot(_jmt.dot(f12.T.dot(_jmt.T))))
            CL = numpy.dot(mm.dot(C) + g*m, _jmt.dot(f12.T.dot(_jmt.T)))
            return numpy.array([alpha_a, alpha_b]),\
                   numpy.array([beta_a, beta_b]),\
                   numpy.array([mu_a, mu_b]),\
                   gamma, AL, BL, CL

        M = m44[:2, :2]
        N = m44[2:, 2:]
        m = m44[:2, 2:]
        n = m44[2:, :2]
        H = m + _jmt.dot(n.T.dot(_jmt.T))
        t = numpy.trace(M - N)
        t2 = t * t
        t2h = t2 + 4.0 * numpy.linalg.det(H)
        g2 = (1.0 + sqrt(t2 / t2h)) / 2
        g = sqrt(g2)
        C = -H * numpy.sign(t) / (g * sqrt(t2h))
        A = g2*M - g*(m.dot(_jmt.dot(C.T.dot(_jmt.T))) + C.dot(n)) + \
            C.dot(N.dot(_jmt.dot(C.T.dot(_jmt.T))))
        B = g2*N + g*(_jmt.dot(C.T.dot(_jmt.T.dot(m))) + n.dot(C)) + \
            _jmt.dot(C.T.dot(_jmt.T.dot(M.dot(C))))
        alpha0_a, beta0_a, tune_a = _closure(A)
        alpha0_b, beta0_b, tune_b = _closure(B)
        inival = (numpy.array([alpha0_a, alpha0_b]),
                  numpy.array([beta0_a, beta0_b]),
                  numpy.zeros((2,)), g, A, B, C)
        rv = (grp1(ms44) for ms44 in mstack)
        return numpy.array([tune_a, tune_b]), inival, rv

    analyze = _analyze4 if coupled else _analyze2
    beamdata, data0, datas = _linopt(ring, dp=dp, refpts=refpts,
                                     get_chrom=get_chrom, get_w=get_w,
                                     analyze=analyze, **kwargs)
    dtype = _DATA1_DTYPE + _W_DTYPE if get_w else _DATA1_DTYPE
    elemdata0 = numpy.array(tuple(data0), dtype=dtype).view(numpy.recarray)
    elemdata = numpy.fromiter(datas, dtype, count=ring.refcount(refpts)
                              ).view(numpy.recarray)
    _unwrap(elemdata.mu)
    return elemdata0, beamdata.tune, beamdata.chromaticity, elemdata


# noinspection PyPep8Naming
@check_radiation(False)
def avlinopt(ring, dp=0.0, refpts=None, **kwargs):
    """
    Perform linear analysis of a lattice and returns average beta, dispersion
    and phase advance

    lindata,avebeta,avemu,avedisp,tune,chrom = avlinopt(ring, dp[, refpts])

    PARAMETERS
        ring            lattice description.
        dp=0.0          momentum deviation.
        refpts=None     elements at which data is returned. It can be:
                        1) an integer in the range [-len(ring), len(ring)-1]
                           selecting the element according to python indexing
                           rules. As a special case, len(ring) is allowed and
                           refers to the end of the last element,
                        2) an ordered list of such integers without duplicates,
                        3) a numpy array of booleans of maximum length
                           len(ring)+1, where selected elements are True.

    KEYWORDS
        orbit           avoids looking for the closed orbit if is already known
                        ((6,) array)
        keep_lattice    Assume no lattice change since the previous tracking.
                        Defaults to False
        XYStep=1.0e-8   transverse step for numerical computation
        DPStep=1.0E-8   momentum deviation used for computation of
                        chromaticities and dispersion
        coupled=True    if False, simplify the calculations by assuming
                        no H/V coupling

    OUTPUT
        lindata         linear optics at the points refered to by refpts, if
                        refpts is None an empty lindata structure is returned.
                        See linopt for details
        avebeta         Average beta functions [betax,betay] at refpts
        avemu           Average phase advances [mux,muy] at refpts
        avedisp         Average dispersion [Dx,Dx',Dy,Dy',muy] at refpts
        avespos         Average s position at refpts
        tune            [tune_A, tune_B], linear tunes for the two normal modes
                        of linear motion [1]
        chrom           [ksi_A , ksi_B], chromaticities ksi = d(nu)/(dP/P).
                        Only computed if 'get_chrom' is True


    See also get_twiss,linopt

    """
    def get_strength(elem):
        try:
            k = elem.PolynomB[1]
        except (AttributeError, IndexError):
            k = 0.0
        return k

    def betadrift(beta0, beta1, alpha0, lg):
        gamma0 = (alpha0 * alpha0 + 1) / beta0
        return 0.5 * (beta0 + beta1) - gamma0 * lg * lg / 6

    def betafoc(beta1, alpha0, alpha1, k2, lg):
        gamma1 = (alpha1 * alpha1 + 1) / beta1
        return 0.5 * ((gamma1 + k2 * beta1) * lg + alpha1 - alpha0) / k2 / lg

    def dispfoc(dispp0, dispp1, k2, lg):
        return (dispp0 - dispp1) / k2 / lg

    boolrefs = bool_refpts([] if refpts is None else refpts, len(ring))
    length = numpy.array([el.Length for el in ring[boolrefs]])
    strength = numpy.array([get_strength(el) for el in ring[boolrefs]])
    longelem = bool_refpts([], len(ring))
    longelem[boolrefs] = (length != 0)

    shorti_refpts = (~longelem) & boolrefs
    longi_refpts = longelem & boolrefs
    longf_refpts = numpy.roll(longi_refpts, 1)

    all_refs = shorti_refpts | longi_refpts | longf_refpts
    _, tune, chrom, d_all = linopt(ring, dp=dp, refpts=all_refs,
                                   get_chrom=True, **kwargs)
    lindata = d_all[boolrefs[all_refs]]

    avebeta = lindata.beta.copy()
    avemu = lindata.mu.copy()
    avedisp = lindata.dispersion.copy()
    aves = lindata.s_pos.copy()

    di = d_all[longi_refpts[all_refs]]
    df = d_all[longf_refpts[all_refs]]

    long = (length != 0.0)
    kfoc = (strength != 0.0)
    foc = long & kfoc
    nofoc = long & (~kfoc)
    K2 = numpy.stack((strength[foc], -strength[foc]), axis=1)
    fff = foc[long]
    length = length.reshape((-1, 1))

    avemu[long] = 0.5 * (di.mu + df.mu)
    aves[long] = 0.5 * (df.s_pos + di.s_pos)
    avebeta[nofoc] = \
        betadrift(di.beta[~fff], df.beta[~fff], di.alpha[~fff], length[nofoc])
    avebeta[foc] = \
        betafoc(df.beta[fff], di.alpha[fff], df.alpha[fff], K2, length[foc])
    avedisp[numpy.ix_(long, [1, 3])] = \
        (df.dispersion[:, [0, 2]] - di.dispersion[:, [0, 2]]) / length[long]
    idx = numpy.ix_(~fff, [0, 2])
    avedisp[numpy.ix_(nofoc, [0, 2])] = (di.dispersion[idx] +
                                         df.dispersion[idx]) * 0.5
    idx = numpy.ix_(fff, [1, 3])
    avedisp[numpy.ix_(foc, [0, 2])] = \
        dispfoc(di.dispersion[idx], df.dispersion[idx], K2, length[foc])
    return lindata, avebeta, avemu, avedisp, aves, tune, chrom


@check_radiation(False)
def get_mcf(ring, dp=0.0, keep_lattice=False, **kwargs):
    """Compute momentum compaction factor

    PARAMETERS
        ring            lattice description
        dp              momentum deviation. Defaults to 0

    KEYWORDS
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
    return (b[5, 1] - b[5, 0]) / dp_step / ring_length[0]


@check_radiation(False)
def get_tune(ring, method='linopt', dp=0.0, **kwargs):
    """
    gets the tune using several available methods

    method can be 'linopt', 'fft' or 'laskar'
    linopt: returns the tune from the linopt function
    fft: tracks a single particle (one for x and one for y)
    and computes the tune from the fft
    laskar: tracks a single particle (one for x and one for y)
    and computes the harmonic components


    INPUT
    for linopt:
        no input needed

    for harmonic:
        nturns: number of turn
        amplitude: amplitude of oscillation
        method: laskar or fft
        num_harmonics: number of harmonic components to compute
        (before mask applied, default=20)
        fmin/fmax: determine the boundaries within which the tune is
        located [default 0->1]
        hann: flag to turn on hanning window [default-> False]
        remove_dc: Removes the mean offset of oscillation data

    OUTPUT
        tunes = np.array([Qx,Qy])
    """

    # noinspection PyShadowingNames
    def gen_centroid(ring, ampl, nturns, dp, remove_dc):
        orbit, _ = find_orbit4(ring, dp)
        ld, _, _, _ = linopt(ring, dp, orbit=orbit)

        invx = numpy.array([[1/numpy.sqrt(ld['beta'][0]), 0],
                            [ld['alpha'][0]/numpy.sqrt(ld['beta'][0]),
                            numpy.sqrt(ld['beta'][0])]])

        invy = numpy.array([[1/numpy.sqrt(ld['beta'][1]), 0],
                            [ld['alpha'][1]/numpy.sqrt(ld['beta'][1]),
                            numpy.sqrt(ld['beta'][1])]])

        p0 = numpy.array([orbit, ]*2).T
        p0[0, 0] += ampl
        p0[2, 1] += ampl
        p1 = lattice_pass(ring, p0, refpts=len(ring), nturns=nturns)
        cent_x = p1[0, 0, 0, :]
        cent_xp = p1[1, 0, 0, :]
        cent_y = p1[2, 1, 0, :]
        cent_yp = p1[3, 1, 0, :]
        if remove_dc:
            cent_x -= numpy.mean(cent_x)
            cent_y -= numpy.mean(cent_y)
            cent_xp -= numpy.mean(cent_xp)
            cent_yp -= numpy.mean(cent_yp)

        cent_x, cent_xp = numpy.matmul(invx, [cent_x, cent_xp])
        cent_y, cent_yp = numpy.matmul(invy, [cent_y, cent_yp])
        return (cent_x - 1j * cent_xp,
                cent_y - 1j * cent_yp)

    if method == 'linopt':
        _, tunes, _, _ = linopt(ring, dp=dp)
    else:
        nturns = kwargs.pop('nturns', 512)
        ampl = kwargs.pop('ampl', 1.0e-6)
        remove_dc = kwargs.pop('remove_dc', True)
        cent_x, cent_y = gen_centroid(ring, ampl, nturns, dp, remove_dc)
        cents = numpy.vstack((cent_x, cent_y))
        tunes = get_tunes_harmonic(cents, method, **kwargs)
    return tunes


@check_radiation(False)
def get_chrom(ring, method='linopt', dp=0, **kwargs):
    """
    gets the chromaticity using several available methods

    method can be 'linopt', 'fft' or 'laskar'
    linopt: returns the chromaticity from the linopt function
    fft: tracks a single particle (one for x and one for y)
    and computes the tune from the fft
    harmonic: tracks a single particle (one for x and one for y)
    and computes the harmonic components

    see get_tune for kwargs inputs

    OUTPUT
        chromaticities = np.array([Q'x,Q'y])
    """

    dp_step = kwargs.pop('DPStep', DConstant.DPStep)
    if method == 'fft':
        print('Warning fft method not accurate to get the ' +
              'chromaticity')

    tune_up = get_tune(ring, method=method, dp=dp + 0.5 * dp_step, **kwargs)
    tune_down = get_tune(ring, method=method, dp=dp - 0.5 * dp_step, **kwargs)
    chrom = (tune_up - tune_down) / dp_step
    return numpy.array(chrom)


Lattice.linopt = linopt
Lattice.linopt2 = linopt2
Lattice.linopt4 = linopt4
Lattice.avlinopt = avlinopt
Lattice.get_mcf = get_mcf
Lattice.avlinopt = avlinopt
Lattice.get_tune = get_tune
Lattice.get_chrom = get_chrom
