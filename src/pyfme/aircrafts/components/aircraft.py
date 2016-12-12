"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

from pyfme.environment.environment import Environment
from pyfme.aircrafts.aircraft import Component
import numpy as np

class Aircraft(Component):
    """The aircraft is a slightly special component, because it is the main
    component, and therefore it has not a partner node.

    Usually Aircraft class is just used as main node to group some subcomponents
    like the wings, propellers, flaps...
    """
    def __init__(self, cog, mass, inertia, Sw):
        """Create a new component

        Parameters
        ----------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of partner- of the center of gravity of this element (m, m, m)
        mass : float
            Mass of the component (kg)
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft.
            Current equations assume that the global aircraft has a symmetry
            plane (x_b - z_b), thus J_xy and J_yz must be null
        Sw : float
            Wetted surface (m2)
        """
        super().__init__(cog, mass, inertia, Sw)

        # Velocities
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number

        # Angular velocities
        self.p = 0  # rad/s
        self.q = 0  # rad/s
        self.r = 0  # rad/s

        # Angles
        self.alpha = 0  # rad
        self.beta = 0  # rad
        self.alpha_dot = 0  # rad/s

    @property
    def q_inf(self):
        """Get the considered dynamic pressure at infinity (Pa),
        :math:`\\frac{1}{2} \\rho \\TAS^2`

        This method is required by 

        Returns
        -------
        q_inf : float
            Sonsidered dynamic pressure at infinity (Pa)
        """
        return 0.5 * environment.rho * self.TAS**2

    def calculate_forces_and_moments(self):
        """Compute the forces and moments of the global aircraft collecting all
        the subcomponents, and adding the volumetric/gravity force
        """
        f, m = super().calculate_forces_and_moments()
        f += environment.gravity_vector * self.mass()
        return f, m
