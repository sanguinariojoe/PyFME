"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

from pyfme.environment.environment import Environment
from pyfme.aircrafts import Component, Controller
from pyfme.aircrafts.components import Aircraft
import numpy as np
from scipy.interpolate import RectBivariateSpline
import re
import inspect


CF_NAMES = "CD", "CY", "CL", "Cl", "Cm", "Cn"
CF_TYPES = ["f"] * 3 + ["m"] * 3


class Wing(Component):
    """A wing. To describe a wing you should provide at least the chord, span
    and wetted surface, as well as the chord and span directions ([1,0,0] and
    [0,1,0]).

    The usual way to use the wing is manually providing the force coefficients.
    However, this tool can be used as well as an abstract class to create a more
    sophisticate method, e.g. a CFD based forces computation.

    Along this line, in order to compute the moments you can either set the wing
    cog as null vector, i.e. the same center of gravity of the partner component
    is considered, conveniently setting the moment coefficients, or you can
    alternatively set the actual forces application point such that the moments
    will be automatically computed.

    This class may not work alone, but it should be a child of an Aircraft.
    Otherwise an assertion error will be raised.
    """
    def __init__(self, chord, span, Sw, parent,
                 chord_vec=np.asarray([1, 0, 0]),
                 span_vec=np.asarray([0, 1, 0]),
                 cog=np.zeros(3, dtype=np.float),
                 mass=0.0,
                 inertia=np.zeros((3, 3), dtype=np.float)):
        """Create a new wing

        Parameters
        ----------
        chord : float
            Wing chord (m)
        span : float
            Wing span (m)
        Sw : float
            Wetted surface (m2)
        parent : Component
            Parent component which owns the current component.
        chord_vec : array_like
            Direction of the wing chord
        span_vec : array_like
            Direction of the wing span
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the considered center
            of the aircraft- of the center of gravity (m, m, m)
        mass : float
            Mass of the component (kg)
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft.
            Current equations assume that the global aircraft has a symmetry
            plane (x_b - z_b), thus J_xy and J_yz must be null
        """
        super().__init__(cog, mass, inertia, Sw, parent=parent)

        # Velocities
        self.__chord = chord
        self.__span = span
        self.__dir = np.array([chord_vec,
                               span_vec,
                               np.cross(chord_vec, span_vec)])
        self.__Cf = []

    @property
    def chord(self):
        """Wing chord (m)

        Returns
        -------
        chord : float
            Wing chord (m)
        """
        return self.__chord

    @chord.setter
    def chord(self, chord):
        """Set the wing chord (m)

        Parameters
        ----------
        chord : float
            Wing chord (m)
        """
        self.__chord = chord

    @property
    def span(self):
        """Wing span (m)

        Returns
        -------
        span : float
            Wing span (m)
        """
        return self.__span

    @chord.setter
    def span(self, span):
        """Set the wing span (m)

        Parameters
        ----------
        span : float
            Wing span (m)
        """
        self.__span = span

    @property
    def chord_vec(self):
        """Chord direction vector

        Returns
        -------
        chord_vec : array_like
            Direction of the wing span
        """
        return self.__dir[0]

    @chord.setter
    def chord_vec(self, span_vec):
        """Set the chord direction vector

        Parameters
        ----------
        chord_vec : array_like
            Direction of the wing span
        """
        self.__dir = np.array([chord_vec,
                               self.__dir[1],
                               np.cross(chord_vec, self.__dir[1])])

    @property
    def span_vec(self):
        """Span direction vector

        Returns
        -------
        span_vec : array_like
            Direction of the wing span
        """
        return self.__dir[1]

    @chord.setter
    def span_vec(self, span_vec):
        """Set the span direction vector

        Parameters
        ----------
        span_vec : array_like
            Direction of the wing span
        """
        self.__dir = np.array([self.__dir[0],
                               span_vec,
                               np.cross(self.__dir[0], span_vec)])

    def add_force_coeff(self, alpha, Cf, name):
        """Register a new force coefficient. The force coefficients are defined
        respect to the cog, and are undimensionalized with the total wetted
        area, i.e. the wetted are of this component and all the children.

        In order to register a force coefficient a variable name should be
        provided, which may be one of the following:

        CL(alpha)
            Lift force coefficient as a function of the angle of attack.
            Lift force acts along ``np.cross(chord_vec, span_vec)``
        CY(alpha)
            Lateral force coefficient as a function of the angle of attack.
            Lateral force acts along ``span_vec``, and depends on the sideslip
            angle, ``beta``, of the aircraft
        CD(alpha)
            Drag force coefficient as a function of the angle of attack.
            Drag is a negative force along ``chord_vec``
        Cl(alpha)
            Roll moment coefficient as a function of the angle of attack.
            Roll moment acts along ``chord_vec``, and depends on the sideslip
            angle, ``beta``, of the aircraft
        Cm(alpha)
            Pitch moment coefficient as a function of the angle of attack.
            Pitch moment acts along ``span_vec``
        Cn(alpha)
            Yaw moment coefficient as a function of the angle of attack.
            Yaw moment acts along ``np.cross(chord_vec, span_vec)``, and depends
            on the sideslip angle, ``beta``, of the aircraft
        CL(alpha, param)
            Lift force coefficients as a function of the angle of attack, due
            to the selected parameter ``param``.
            Lift force acts along ``np.cross(chord_vec, span_vec)``
        CY(alpha, param)
            Lateral force coefficients as a function of the angle of attack, due
            to the selected parameter ``param``.
            Lateral force acts along ``span_vec``
        CD(alpha, param)
            Drag force coefficients as a function of the angle of attack, due
            to the selected parameter ``param``.
            Drag is a negative force along ``chord_vec``
        Cl(alpha, param)
            Roll moment coefficient as a function of the angle of attack, due
            to the selected parameter ``param``.
            Roll moment acts along ``chord_vec``
        Cm(alpha, param)
            Pitch moment coefficient as a function of the angle of attack, due
            to the selected parameter ``param``.
            Pitch moment acts along ``span_vec``
        Cn(alpha, param)
            Yaw moment coefficient as a function of the angle of attack, due
            to the selected parameter ``param``.
            Yaw moment acts along ``np.cross(chord_vec, span_vec)``

        The available parameters ``param`` are ``p``, ``q``, ``r`` and
        ``alphadot``, the rate of change of Roll, Pitch, Yaw, and angles of
        attack respectively.

        Parameters
        ----------
        alpha : array_like
            List of considered values for the angle of attack to interpolate the
            force coefficients. If this is None, all the force coefficients are
            ignored, which may be convenient to overload this class with a more
            complex stuff.
        Cf : array_like
            Force coefficients as a function of the angle of attack. This should
            have the same length than ``alpha``
        name : string
            Force coeffcient name, including its dependency parameters
        """
        params = "p", "q", "r", "alphadot"
        def get_available_names():
            l = []
            for n in CF_NAMES:
                l.append(n + "(alpha)")
                for p in params:
                    l.append(n + "(alpha, " + p + ")")
            return l

        # Get the variable name and parameters
        pattern = r'(\w[\w\d_]*)\((.*)\)$'
        match = re.match(pattern, name.replace(" ", ""))
        if not match:
            raise ValueError("Can't parse expression '{}'".format(name))
        groups = match.groups()
        if len(groups) != 2:
            raise ValueError(
                "Can't split expression '{}' in name and params".format(name))
        cname = groups[0]
        params = groups[1].split(',')
        if name not in names or len(params) > 2 or 'alpha' not in params:
            raise ValueError(
                "Invalid name '{}'. The valid ones are: {}".format(
                    name, get_available_names()))

        param = params[params.index("alpha") - 1] if len(params) == 2 else None

        self.__Cf.append({'alphas':np.asarray(alpha),
                          'values'np.asarray(Cf),
                          'name':cname,
                          'param':param})

    def __get_coeff_type(self, Cf):
        """Report whether it is a force coefficient or a moment coefficient

        Parameters
        ----------
        Cf : dictionary
            Force/Moment coefficient to parse

        Returns
        -------
        type_coeff : string
            'f' if it is a force coefficient, 'm' otherwise
        """
        CF_NAMES = "CD", "CY", "CL", "Cl", "Cm", "Cn"
        return CF_TYPES[CF_NAMES.index(Cf[3])]

    def __get_coeff_vec(self, Cf):
        """Get the direction of an specific force coefficient by its name

        Parameters
        ----------
        Cf : dictionary
            Force coefficient to parse

        Returns
        -------
        vec : array_like
            Direction of the force coefficient
        """
        CF_NAMES = "CD", "CY", "CL", "Cl", "Cm", "Cn"
        return self.__dir[CF_NAMES.index(Cf[3]) - 3]

    def _raw_coeff(self, Cf, alpha):
        """Interpolate the Force coefficient raw value, this is, before
        modifying it by the sideslip angle, or other parameters.

        This method is conveniently provided to make easier to overload the
        ``Flap`` class, where ``alpha`` and the controller value should be used.

        Parameters
        ----------
        Cf : dictionary
            Force coefficient to parse
        alpha : float
            Angle of attack (deg)

        Returns
        -------
        c : float
            Raw force/moment coefficient
        """
        return np.interp(alpha, Cf['alphas'], Cf['values'])

    def __solve_coeff(self, Cf, alpha, beta, V, p, q, r, alphadot):
        """Get the Force coeffcient vectorial value, i.e. with orientation
        implicitely included.

        Parameters
        ----------
        Cf : dictionary
            Force coefficient to parse
        alpha : float
            Angle of attack (deg)
        beta : float
            Slideslip angle (deg)
        V : float
            True Air Speed (m/s)
        p : float
            Roll rate of change (rad/s)
        q : float
            Pitch rate of change (rad/s)
        r : float
            Yaw rate of change (rad/s)
        alphadot : float
            Angle of attack rate of change (rad/s)

        Returns
        -------
        vec : array_like
            Direction of the force coefficient
        """
        # First, let's interpolate the value of the raw coefficient
        c = _raw_coeff(Cf, alpha)
        # Get the force direction
        vec = self.__get_coeff_vec(Cf)
        # Multiply the coeffcient by the required parameters
        if Cf['param'] is None:
            if Cf['name'] in ('CY', 'Cl', 'Cn'):
                # Special case of transversal force, roll moment and yaw moment,
                # which depends on the sideslip angle
                c *= beta
        else:
            # Get the characteristic length (chord or span)
            d = np.dot((self.span, self.span, self.chord), vec)
            # Get the required paramater
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            param = values[args.index(Cf['param'])]
            # And get the final equivalent force coeff
            c *= d / (2 * V) * param

        # Return it as a vector
        return c * vec

    def calculate_forces_and_moments(self):
        """Compute the forces and moments of the global aircraft collecting all
        the subcomponents

        Returns
        -------
        f : array_like
            Drag, lateral and Lift forces (N)
        m : array_like
            Roll, pitch and yaw moments (N * m)
        """
        f, m = super().calculate_forces_and_moments()

        if self.__alpha is None or not len(self.__alpha):
            # Special case where the user don't want to use the coeffs stuff
            return f, m

        # Get the aircraft data
        aircraft = self.top_node()
        assert isinstance(aircraft, Aircraft)
        attack_angles = np.asarray(0.0,
                                   aircraft.alpha,  # rad
                                   aircraft.beta)   # rad
        attack_angles_dot = np.asarray(0.0,
                                       aircraft.alphadot,  # rad/s
                                       aircraft.betadot)   # rad/s
        p = aircraft.p  # rad/s
        q = aircraft.q  # rad/s
        r = aircraft.r  # rad/s
        # FIXME: Vectorial velocities should be considered to can model other
        # aircraft types, like helicopters
        V = aircraft.TAS  # m/s
        q_inf = aircraft.q_inf  # Pa

        # Get the angle of attack (the angle along the span direction)
        alpha = np.rad2deg(np.dot(attack_angles, self.__dir[1]))  # deg
        alphadot = np.dot(attack_angles_dot, self.__dir[1])       # rad/s
        # Get the sideslip angle (the angle along the "z" direction)
        beta = np.dot(attack_angles, self.__dir[2])               # rad

        # Compute the forces from the force coefficients
        for Cf in self.__Cf:
            Sw = self.Sw()
            if __get_coeff_type(Cf) == 'f':
                f += q_inf * Sw * __solve_coeff(Cf, alpha, beta,
                                                V, p, q, r, alphadot)
            else self.__get_coeff_type(Cf) == 'm':
                # The moments should be multiplied by the chord or the span
                d = np.dot((self.span, self.chord, self.span),
                           self.__get_coeff_vec(Cf))
                m += d * q_inf * Sw * __solve_coeff(Cf, alpha, beta,
                                                    V, p, q, r, alphadot)

        return f, m


class Flap(Wing):
    """A flap. The flaps are basically like wings, except because:

    * They are intended to be childs of a Wing
    * If the chord, span, wetted surface, or orientation vectors are not
      provided, they are taken from the parent ``Wing`` component.
    * They have a controller
    * The force/moment coefficients should be interpolated from the angle of
      attack, ``alpha``, and the controller angle value. Therefore, the
      Coefficients be a matrix of (m, n) dimensions, where n is the number of
      considered alpha values and m is the number of considered deflection
      angles (the deflection angle is edited by the controller).

    The flaps can be therefore seen as "wing modifiers". Along this line, you
    probably only want to register CL(alpha), CY(alpha), CD(alpha) force
    coefficients and Cl(alpha), Cm(alpha) and Cn(alpha) moment coefficients
    """
    def __init__(self, parent, angles,
                 controller_name='delta_flap',
                 chord=0.0,
                 span=0.0,
                 Sw=0.0,
                 chord_vec=None,
                 span_vec=None,
                 cog=np.zeros(3, dtype=np.float),
                 mass=0.0,
                 inertia=np.zeros((3, 3), dtype=np.float)):
        """Create a new flap

        Parameters
        ----------
        parent : Component
            Parent component which owns the current component. It should be a
            ``Wing``
        angles : float
            List of deflection angles considered for the flap (deg). The
            controller will be set using the minimum and maximum values of this
            parameter
        controller_name : string
            Name of the associated controller to be generated
        chord : float
            Wing chord (m). 0 if it should be extracted from the parent
            component
        span : float
            Wing span (m). 0 if it should be extracted from the parent
            component
        Sw : float
            Wetted surface (m2). 0 if it should be extracted from the parent
            component
        chord_vec : array_like
            Direction of the wing chord. None if it should be extracted from the
            parent component
        span_vec : array_like
            Direction of the wing span. None if it should be extracted from the
            parent component
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the considered center
            of the aircraft- of the center of gravity (m, m, m)
        mass : float
            Mass of the component (kg)
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft.
            Current equations assume that the global aircraft has a symmetry
            plane (x_b - z_b), thus J_xy and J_yz must be null
        """
        super().__init__(chord, span, Sw, parent,
                         chord_vec=chord_vec,
                         span_vec=span_vec,
                         cog=cog,
                         mass=mass,
                         inertia=inertia)

        self.controller = Controller(controller_name,
                                     np.min(angles),
                                     np.max(angles))
        self.__angles = angles()

    @property
    def angles(self):
        """List of deflection angles considered for the flap (deg)

        Returns
        -------
        angles : float
            List of deflection angles considered for the flap (deg). The
            controller will be set using the minimum and maximum values of this
            parameter
        """
        return self.__angles = angles

    @chord.setter
    def angles(self, span_vec):
        """Set the list of deflection angles considered for the flap (deg)

        Parameters
        ----------
        angles : float
            List of deflection angles considered for the flap (deg). The
            controller will be set using the minimum and maximum values of this
            parameter
        """
        self.__angles = angles

    def _raw_coeff(self, Cf, alpha):
        """Interpolate the Force coefficient raw value, this is, before
        modifying it by the sideslip angle, or other parameters.

        Conversely to the wing, where the force/moment coefficient depends only
        on the attack angle ``alpha``, here another dependency on the controller
        value should be considered.

        Hence, the force/moment coefficients be a matrix of (m, n) dimensions,
        where n is the number of considered alpha values and m is the number of
        considered deflection angles, ``self.angles``.

        Parameters
        ----------
        Cf : dictionary
            Force coefficient to parse
        alpha : float
            Angle of attack (deg)

        Returns
        -------
        c : float
            Raw force/moment coefficient
        """
        interpolator = RectBivariateSpline(self.__angles,
                                           Cf['alphas'],
                                           Cf['values'])
        return interpolator(controller.value, alpha)

    def calculate_forces_and_moments(self):
        """Compute the forces and moments of the global aircraft collecting all
        the subcomponents, and adding the volumetric/gravity force

        Returns
        -------
        f : array_like
            Drag, lateral and Lift forces (N)
        m : array_like
            Roll, pitch and yaw moments (N * m)
        """
        wing = self.parent
        # Get the partner parameters
        if not self.chord:
            assert isinstance(wing, Wing)
            self.chord = wing.chord
        if not self.span:
            assert isinstance(wing, Wing)
            self.span = wing.span
        if self.chord_vec is None:
            assert isinstance(wing, Wing)
            self.chord_vec = wing.chord_vec
        if self.span_vec is None:
            assert isinstance(wing, Wing)
            self.span_vec = wing.span_vec
        # The wetted surface is a bit more delicated, because if someone else
        # is asking for the wetted surface, we want to say it is null
        is_Sw_zero = not self.Sw(use_subcomponents=False)
        if is_Sw_zero:
            assert isinstance(wing, Wing)
            super(Structure, self).Sw = wing.Sw(use_subcomponents=False)

        # Compute the forces and moments
        f, m = super(Wing, self).calculate_forces_and_moments()

        # Restore the wetted surface
        if is_Sw_zero:
            assert isinstance(wing, Wing)
            super(Structure, self).Sw = 0.0

        return f, m
