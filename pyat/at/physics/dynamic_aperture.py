import at
import copy
import at.plot
import at.lattice.cavity_access
from at.lattice import AtError
from itertools import compress
import numpy as np
import math
import pickle
from scipy.io import savemat
from scipy.constants import c as clight
import time

import warnings

__all__ = ['DynamicAperture', 'Acceptance6D', 'dynamic_aperture', 'off_energy_dynamic_aperture', 'momentum_acceptance']

class DynamicAperture(object):


    def __init__(self,
                 ring,
                 dp=0.0,
                 mode = ['x','xp'],
                 npoints = [13, 13],
                 ranges = [[-1e-3, 1e-3], [-1e-3, 1e-3]],
                 center_points = np.zeros(6),
                 start_index=0,
                 grid_mode='grid',
                 n_turns=2**10,
                 compute_limits = False,
                 parallel = False,
                 verbose=True
                ):
        """
        Acceptance6D computes the 6D acceptance for a pyAT lattice

        :param ring: pyAT lattice
        :param dp: momentum deviation
        :param mode: mode for computation. None = 6D, '6D', 'x-y' (default), 'delta-x'. 'x-xp',...
        :param start_index: index in ring where to start the computation
        :param grid_mode: 'grid' or 'radial' set of test points to compute Acceptance
        :param compute_range: compute range for each plane with more than 1 point to scan
        :param nturns: number of turns that a particle must survive
        :param search_divider: division of range used by recursive range
        :param parallel: default False if True, use patpass when usefull
        : n_point: dictionary to determine the # points to scan in each dimension
        : dict_def_range: default range for each dimension
        : dict_unit:  default units for each dimension

        """
        #if not compute_limits:
        assert ranges!=None, 'ranges must be defined for grid mode'
        assert np.shape(ranges) == (2,2), 'ranges must be a 2x2 array'
    
        assert npoints, 'npoints must be defined for grid mode'
        assert np.shape(npoints) == (2,), 'npoints must be an array or list with 2 elements'

        assert grid_mode=='grid' or grid_mode=='radial', 'grid_mode must be grid or radial'
        self.mode = mode
        self.ranges = ranges 
        self.npoints = npoints
        self.grid_mode = grid_mode
        self.n_turns = n_turns
        self.center_points = center_points
        self.parallel_computation = parallel
        self.compute_limits = compute_limits
        self.verbose = verbose

        self.planes = np.array(['x', 'xp', 'y', 'yp', 'delta', 'ct'])
        try:
            self.hor_ind = np.where(self.planes == mode[0])[0][0]
            self.ver_ind = np.where(self.planes == mode[1])[0][0]
        except IndexError:
            print('Incorrect mode specified')
            exit()

        self.dict_units_for_plotting = {'x': [1e3, 'mm'],
                                        'xp': [1e3, 'mrad'],
                                        'y': [1e3, 'mm'],
                                        'yp': [1e3, 'mrad'],
                                        'delta': [1e2, '%'],
                                        'ct': [1e3, 'mm'],
                                         }

        # rotate lattice to given element
        self.ring = at.Lattice(ring[start_index:] + ring[:start_index])

        self.dp = dp

        # radiation ON means "there is an active cavity", whether there is radiation or not.
        
        self.mcf = self.ring.radiation_off(copy=True).get_mcf()

        # get RF indexes in lattice
        self.ind_rf = [i for i, elem in enumerate(ring) if isinstance(elem, at.RFCavity)]

        Circumference = self.ring.circumference # s_range[-1] not working if periodicity != 1
        harm = self.ring.get_rf_harmonic_number()  # use pyat/at/utility/cavity_acess.py module
        rf_frequency_0  = harm * clight / Circumference
        self.rf_frequency = self.ring.get_rf_frequency()  # lattice RF

        if abs(self.rf_frequency-rf_frequency_0) > 1e6:
            if self.verbose:
                print('set frequency to ideal one. more than 1MHz difference')
            self.rf_frequency = rf_frequency_0  # if RF not nominal, set to nominal

        # define orbit about wich to compute DA
        self.compute_orbit()

        # define coordinates dictionary structure
        self.coordinates = {'x': [],
                            'xp': [],
                            'y': [],
                            'yp': [],
                            'delta': [],
                            'ct': []
                            }
        self.survived = []

        if self.compute_limits:
            self.compute_range()


    def compute_orbit(self):
        """
        computes orbit for given dp
        """

        if self.ring.radiation:
            # set RF frequency
            drf = - self.rf_frequency * self.mcf * self.dp

            new_rf_frequency = self.rf_frequency + drf
            if self.verbose:
                print('dp = {dp}, rf set to f_RF0 + {rf} Hz'.format(dp=self.dp, rf=drf))
                print('new RF = {rf:3.6f} MHz'.format(rf=new_rf_frequency*1e-6))

            # set RF frequency to all cavities
            try :
                self.ring = self.ring.set_rf_frequency(new_rf_frequency, copy=True)
            except AtError as exc:
                print('RF not set. Probably the lattice has no RFCavity.')
                raise(exc)

            # compute orbit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #disable this warning: AtWarning: In 6D, "dp" and "dct" are ignored warnings.warn(AtWarning('In 6D, "dp" and "dct" are ignored'))
                self.orbit, _ = self.ring.find_orbit(dp=self.dp)  # set dp even if ignored to be used if rad is off

        else:
            if self.verbose:
                print('no cavity in the lattice, using find_orbit4')

            rad = self.ring.radiation
            self.ring.radiation_off()

            self.orbit, _ = self.ring.find_orbit4(dp=self.dp)  # dpp is added to orbit here

            if rad:
                self.ring.radiation_on()


        return self.orbit

    def compute_range(self, factor=1.2):
        """
        computes maximum +/- range for all dimensions with more than 1 point to scan
        result is stored in dict_def_range
        ranges are limited to [-1 1] in any case without notice.
        :return:
        """
        # disable parallel computation
        init_parcomp = self.parallel_computation
        self.parallel_computation = False
    
        self.ranges = [[-1e-3,1e-3],[-1e-3,1e-3]]

        direction = np.zeros(6)
        direction[self.hor_ind] = -1
        self.find_limits(direction=direction)
        self.ranges[0][0] = factor*self.limit[self.hor_ind]

        direction = np.zeros(6)
        direction[self.hor_ind] = 1
        self.find_limits(direction=direction)
        self.ranges[0][1] = factor*self.limit[self.hor_ind]

        direction = np.zeros(6)
        direction[self.ver_ind] = -1
        self.find_limits(direction=direction)
        self.ranges[1][0] = factor*self.limit[self.ver_ind]

        direction = np.zeros(6)
        direction[self.ver_ind] = 1
        self.find_limits(direction=direction)
        self.ranges[1][1] = factor*self.limit[self.ver_ind]

        if self.verbose:
            print('Computed range for {pl}: [{mi}, {ma}]'.format(pl=self.planes[self.hor_ind],
                                                                 mi=self.ranges[0][0],
                                                                 ma=self.ranges[0][1]))
            print('Computed range for {pl}: [{mi}, {ma}]'.format(pl=self.planes[self.ver_ind],
                                                                 mi=self.ranges[1][0],
                                                                 ma=self.ranges[1][1]))
        # restore initial parallel computation value
        self.parallel_computation = init_parcomp


    def find_limits(self, direction=(1, 0, 0, 0, 0, 0),
                                              number_of_recursions=3,
                                              back_step=2,
                                              search_divider=3):
        """

        recursively search for maximum coordinate along a given direction where particle survive

        :param direction: 6D array to deterine a given direction
        :param number_of_recursions:  number of recursion. at each recursion, the last step in the search is replaced by a finer mesh
        :param back_step:  number of steps to decrement to start with next search.
        :return: 6D coordinates array of last surviving particle
        """

        # disable parallel computation
        init_parcomp = self.parallel_computation
        self.parallel_computation = False

        # define initial coordinate
        self.coordinates = {'x': [1e-6], 'xp': [0.0], 'y': [1e-6], 'yp': [0.0], 'delta': [0.0], 'ct': [0.0]}
        #coord = np.array([1e-6, 0, 1e-6, 0, 0, 0])
        # define step in each plane
        step = []
        for ip, pl in enumerate(self.planes):
            if ip == self.hor_ind or ip == self.ver_ind:
                if ip == self.hor_ind:
                    ind = 0
                elif ip == self.ver_ind:    
                    ind = 1
                step.append(direction[ip]*(self.ranges[ind][1] - self.ranges[ind][0]))
            else:
                step.append(0)
        step = np.array(step)
        # limit = [v for _, v in coord.items()]
        #limit = list(coord.values())

        for ir in range(number_of_recursions):

            if self.verbose:
                print('search {d} recursively step {s}/{ts}'.format(d=direction, s=ir, ts=number_of_recursions))

            # define initial coordinates for search
            #if ir > 0:
            #    for pl in self.planes:
            #        coord[pl] -= step[pl]  # go back by one of the previous size steps.
            #else:
            #    coord = copy.deepcopy(coord0)
            # if ir == 0:
            #    coord = {'x': 1e-6, 'xp': 0.0, 'y': 1e-6, 'yp': 0.0, 'delta': 0.0, 'ct': 0.0}

            if self.verbose:
                print(self.coordinates)

            # reduce step (initial = full range)
            step  /= search_divider

            # do not recompute already computed point
            if ir > 0:
                for ip, pl in enumerate(self.planes):
                    self.coordinates[pl][0] += step[ip]  # go back by one of the previous size steps.

            # search limit
            self.track_particles()
            while self.survived[0]:
                for ip, pl in enumerate(self.planes):
                    self.coordinates[pl][0] += step[ip] 
                # update tested points
                #self.coordinates[pl].append(coord[pl])
                if self.verbose:
                    print('step forward')
                    print([self.coordinates[key][0] for key in self.coordinates.keys()])
                self.track_particles()

            # last survived is previous point
            for ip, pl in enumerate(self.planes):
                _c = back_step * step[ip]
                # limit to 0.0
                if self.coordinates[pl][0] > 0.0:
                    if self.coordinates[pl][0] - _c < 0.0:
                        _c = self.coordinates[pl][0]
                if self.coordinates[pl][0] < 0.0:
                    if self.coordinates[pl][0] - _c > 0.0:
                        _c = self.coordinates[pl][0]
                # assign new start coord
                self.coordinates[pl][0] -= _c

            if self.verbose:
                print('step back')
                print(self.coordinates)

            # last coordinates are lost and add an additional test point lost (for display and contour computation
            #if self.survived[0]:
            #    self.survived[-1] = False

            coord_lost = copy.deepcopy(self.coordinates)
            for ip, pl in enumerate(self.planes):
                coord_lost[pl] += step[ip]
                # update tested points
                #self.coordinates[pl].append(coord_lost[pl])
            self.survived.append(False)

            # define limit 6 element array
            # limit = [copy.deepcopy(v) for _, v in coord.items()]
            
            self.limit = np.array([self.coordinates[pl][0] for pl in self.planes])
            if self.verbose:
                print('present limit: {l}'.format(l=self.limit))

        # restore initial parallel computation value
        self.parallel_computation = init_parcomp



    def generate_particles(self):
        """
        define fixed grid of test points
        the grid of points is stored in a dictionary of coordinates, one array for each dimension

        :param compute_limits calls compute_range to deterine the maximum ranges before defining the grid

        :return:
        """

        # define grid
        if self.grid_mode == 'radial':

            d_ = {'x': [], 'xp': [], 'y': [], 'yp': [], 'delta': [], 'ct': []}


            ellipse_axes = [self.ranges[0][1], self.ranges[1][1]]

            if np.mean(self.ranges[1]) == 0:
                self.halfrange = True
                theta = np.linspace(0, math.pi, self.npoints[0])
            else:
                self.halfrange = False
                theta = np.linspace(0, 2*math.pi, self.npoints[0])

            ip = 0
            for i, p in enumerate(self.planes):  # must loop all plane to define all 6D ranges (1 point)
                if p == self.mode[0] or p==self.mode[1]:
                    # scan predefined range
                    for ea in np.sqrt(np.linspace(ellipse_axes[ip]**2/self.npoints[1],
                                          ellipse_axes[ip]**2, self.npoints[1])):
                        if ip == 0:
                            d_[p].append([ea * math.cos(t) for t in theta])
                        elif ip == 1:
                            d_[p].append([ea * math.sin(t) for t in theta])

                    ip += 1

                else:  # this dimension is not part of the scan
                    d_[p].append([self.center_points[i]])

            flat_dd = []
            for k, val in d_.items():
                flat_dd.append([item for sublist in val for item in sublist])
                if self.verbose:
                    print('{} has {} elements'.format(k, len(flat_dd[-1])))

            num_points = max([len(dd) for dd in flat_dd])

            # define lists for DA scan
            for p in self.planes:
                if p == self.mode[0] or p==self.mode[1]:
                    self.coordinates[p] = np.array([item for sublist in d_[p] for item in sublist])
                else:
                    self.coordinates[p] = np.array([item for sublist in d_[p] for item in sublist]*num_points)

        else:

            d_ = []
            
            for i, p in enumerate(self.planes):
                ip = 0
                if p == self.mode[0] or p==self.mode[1]:
                    if i == self.hor_ind:
                        ind = 0
                    elif i==self.ver_ind:
                        ind = 1
                    d_.append(np.linspace(self.ranges[ind][0],
                                          self.ranges[ind][1],
                                          self.npoints[ind]))
                else:  # this dimension is not part of the scan
                    d_.append((self.center_points[i]))

            # define mesh of points
            xx, xpxp, yy, ypyp, deltadelta, ctct = np.meshgrid(*d_)

            self.coordinates['x'] = xx.flatten()
            self.coordinates['xp'] = xpxp.flatten()
            self.coordinates['y'] = yy.flatten()
            self.coordinates['yp'] = ypyp.flatten()
            self.coordinates['delta'] = deltadelta.flatten()
            self.coordinates['ct'] = ctct.flatten()

        if len(self.coordinates['x']) == 0:
            raise IOError('grid mode must be grid or radial')

        self.survived = [False]*len(self.coordinates['x'])

        pass

    def track_particles(self):
        """
        test if a set of particle coordinates survived
        returns a boolean if the coordinates survived
        """

        # nothing to do if no coordinate to test
        if len(self.coordinates) == 0:
            return []

        # if coordinates to test:


        # create 6xN matrix add orbit (with dpp) to each coordinate
        rin = np.concatenate(([self.coordinates['x'] + self.orbit[0]],
                              [self.coordinates['xp'] + self.orbit[1]],
                              [self.coordinates['y'] + self.orbit[2]],
                              [self.coordinates['yp'] + self.orbit[3]],
                              [self.coordinates['delta'] + self.orbit[4]],
                              [self.coordinates['ct'] + self.orbit[5]]),
                              axis=0)

        rin = np.asfortranarray(rin)

        # track coordinates for N turns
        if not self.parallel_computation:

            t = at.lattice_pass(self.ring,
                          rin.copy(),
                          self.n_turns)

        else:
            if self.verbose:
                print('parallel computation')

            # track coordinates
            t = at.patpass(self.ring,
                           rin.copy(),
                           self.n_turns,
                           refpts=np.array(np.uint32(0)))

        self.survived = [not s for s in np.isnan(t[0,:,0,-1])]

        # print if survived for each test particle
        if self.verbose:
            if type(self.coordinates['x']) is float:
                print('[{x:2.2g}, {xp:2.2g}, {y:2.2g}, {yp:2.2g}, {delta:2.2g}, {ct:2.2g}] '
                      '| {tot:02d} coord: {surv} for {nt} turns'.format(
                    tot=1, surv=self.survived, nt=self.n_turns,
                    x=self.coordinates['x'], y=cself.oordinates['y'], xp=self.coordinates['xp'],
                    ct=self.coordinates['ct'], yp=self.coordinates['yp'], delta=self.coordinates['delta']))
            else:
                for x, xp, y, yp, delta, ct, s in zip(self.coordinates['x'],
                                                      self.coordinates['xp'],
                                                      self.coordinates['y'],
                                                      self.coordinates['yp'],
                                                      self.coordinates['delta'],
                                                      self.coordinates['ct'], self.survived):
                    print('[{x:2.2g}, {xp:2.2g}, {y:2.2g}, {yp:2.2g}, {delta:2.2g}, {ct:2.2g}] '
                          '| {tot:02d} coord: {surv} for {nt} turns'.format(
                        tot=len(self.coordinates['x']), surv=s, nt=self.n_turns,
                        x=x, y=y, xp=xp, ct=ct, yp=yp, delta=delta))
        

    def compute_DA(self):
        """
        compute 2D DA limits

        fills the self.survived property with booleans corresponding to each coordinate in coordinates

        :return: h_s, v_s, (h), (v), (survived) lists of dim1 and dim2 coordinates defining the 2D limit of Acceptance
        :return: (h_s), (v_s), h, v, survived all coordinates tested and array of survival reduced to the 2D in self.mode
        """

        # display summary of work to do
        if self.verbose:
            print('Computing acceptance {m1}-{m2}'.format(m1=self.mode[0], m2=self.mode[1]))


        h_border = []
        v_border = []
        h_all = []
        v_all = []
        survived = []

        # simplify 6D structure to 2D
        h_all, v_all, survived = self.get_relevant_points_for_DA_computation()

        # find maximum of each column and return as border
        try:
            h_border, v_border = self.get_border(h_all, v_all, survived)
        except Exception:
            if self.verbose:
                print('DA limit could not be computed (probably no closed contour)')



        return h_border, v_border, h_all, v_all, survived

    def get_border(self, h, v, sel):
        """
        find border of list of points.
        works only for 2D and grid mode.
        """
        h_border = []
        v_border = []

        # [print(h_, v_, s_) for h_, v_, s_ in zip(h, v, sel)]

        # loop columns of grid
        if self.grid_mode == 'grid':

            for hc in np.unique(h):
                col = []
                notcol = []
                for i, h_ in enumerate(h):

                    if h_ == hc and sel[i]:
                        col.append(v[i])
                    elif h_ == hc and not sel[i]:
                        notcol.append(v[i])

                if len(col) > 0:

                    # compute step size from grid
                    if len(col)>1:
                        step = col[-1]-col[-2]
                    elif len(notcol)>1:
                        step = notcol[-1] - notcol[-2]

                    # append a point for the first lost on the column from above
                    h_border.append(hc)
                    v_border.append(np.min(notcol)-step)
                    # appen a point on for the firt lost on the column from below
                    h_border.insert(0,hc)
                    v_border.insert(0,np.min(col))

        elif self.grid_mode == 'radial':
            # find radial grid extremes.  not implemented
            print('radial mode, no border computed')
            pass
        else:
            print('grid_mode must be grid or radial')

        # remove bottom line if any. not implemented

        return h_border, v_border

    def get_relevant_points_for_DA_computation(self):
        """
        reduces the 6D coordinated according to the mode to 3 lists for plotting:
        horizontal, vertical and boolean array of survival.
        """

        if self.mode == None or self.mode=='6D':
            if self.verbose:
                print('no selection of test points for modes None and 6D')
            return [], [], []

        if self.mode == 'x-y':
            h = self.coordinates['x']
            v = self.coordinates['y']

            sel = [a and b == 0 and c == 0 and d == 0 and e == 0 for a, b, c, d, e in zip(
                self.survived,
                self.coordinates['delta'],
                self.coordinates['xp'],
                self.coordinates['yp'],
                self.coordinates['ct'])]

        h = self.coordinates[self.mode[0]]
        v = self.coordinates[self.mode[1]]
        otherplanes =[]
        [otherplanes.append(plane) for plane in self.planes if self.mode[0]!=plane and self.mode[1]!= plane]
        sel = [a and b == 0 and c == 0 and d == 0 and e == 0 for a, b, c, d, e in zip(
            self.survived,
            self.coordinates[otherplanes[0]],
            self.coordinates[otherplanes[1]],
            self.coordinates[otherplanes[2]],
            self.coordinates[otherplanes[3]])]

        return h, v, sel


