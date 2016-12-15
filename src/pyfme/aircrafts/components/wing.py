"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

from pyfme.environment.environment import Environment
from pyfme.aircrafts import Component, Controller
from pyfme.aircrafts.components import Aircraft
import numpy as np


class Wing(Component):
    """A wing. To describe a wing you should provide at least the chord, span
    and wetted surface, as well as the chord and span directions ([1,0,0] and
    [0,1,0]).

    The usual way to use the wing is manually providing the force coefficients.
    However, this tool can be used as well as an abstract class to create a more
    sophisticate method, e.g. a CFD based forces computation.
    If a force coefficient is not provided, it will be considered null.

    This class may not work alone, but it should be a child of an Aircraft.
    Otherwise an assertion error will be raised.
    """
    def __init__(self, chord, span, Sw, parent,
                 chord_vec=np.asarray([1, 0, 0]),
                 span_vec=np.asarray([0, 1, 0]),
                 alpha=None,
                 CL=None,
                 CY=None,
                 CD=None,
                 Cl=None,
                 Cm=None,
                 Cn=None,
                 CL_q=None,
                 CY_q=None,
                 CD_q=None,
                 Cl_q=None,
                 Cm_q=None,
                 Cn_q=None,
                 CL_alphadot=None,
                 CY_alphadot=None,
                 CD_alphadot=None,
                 Cl_alphadot=None,
                 Cm_alphadot=None,
                 Cn_alphadot=None,
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
        alpha : array_like
            List of considered values for the angle of attack to interpolate the
            force coefficients
        CL : array_like
            Lift force coefficients as a function of the angle of attack.
            Lift force acts along ``np.cross(chord_vec, span_vec)``
        CY : array_like
            Lateral force coefficients as a function of the angle of attack.
            Lateral force acts along ``span_vec``, and depends on the yaw angle,
            ``beta``, of the aircraft
        CD : array_like
            Drag force coefficients as a function of the angle of attack.
            Drag is a negative force along ``chord_vec``
        CL_q : array_like
            Lift force coefficients as a function of the angle of attack, due to
            the Pitch rate of change.
            Lift force acts along ``np.cross(chord_vec, span_vec)``
        CY_q : array_like
            Lateral force coefficients as a function of the angle of attack, due
            to the Pitch rate of change.
            Lateral force acts along ``span_vec``, and depends on the yaw angle,
            ``beta``, of the aircraft
        CD_q : array_like
            Drag force coefficients as a function of the angle of attack, due to
            the Pitch rate of change.
            Drag is a negative force along ``chord_vec``
        CL_alphadot : array_like
            Lift force coefficients as a function of the angle of attack, due to
            the angle of attack rate of change.
            Lift force acts along ``np.cross(chord_vec, span_vec)``
        CY_alphadot : array_like
            Lateral force coefficients as a function of the angle of attack, due
            to the angle of attack rate of change.
            Lateral force acts along ``span_vec``, and depends on the yaw angle,
            ``beta``, of the aircraft
        CD_alphadot : array_like
            Drag force coefficients as a function of the angle of attack, due to
            the angle of attack rate of change.
            Drag is a negative force along ``chord_vec``
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
        self.__alpha = alpha
        self.__CL = CL
        self.__CY = CY
        self.__CD = CD
        self.__CL_q = CL_q
        self.__CY_q = CY_q
        self.__CD_q = CD_q
        self.__CL_alphadot = CL_alphadot
        self.__CY_alphadot = CY_alphadot
        self.__CD_alphadot = CD_alphadot

    def calculate_forces_and_moments(self):
        """Compute the forces and moments of the global aircraft collecting all
        the subcomponents, and adding the volumetric/gravity force
        """
        f, m = super().calculate_forces_and_moments()
        return f, m
