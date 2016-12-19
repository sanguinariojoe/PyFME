"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

from pyfme.environment.environment import Environment
from pyfme.aircrafts import Component
from pyfme.utils.anemometry import tas2cas, tas2eas, calculate_alpha_beta_TAS
import numpy as np


class Aircraft(Component):
    """The aircraft is a slightly special component, because it is the main
    component, and therefore it has not a parent node.

    Usually Aircraft class is just used as main node to group some subcomponents
    like the hull, wings, propellers, flaps...
    """
    def __init__(self,
                 cog=np.zeros(3, dtype=np.float),
                 mass=0.0,
                 inertia=np.zeros((3, 3), dtype=np.float),
                 Sw=0.0):
        """Create a new aircraft

        Parameters
        ----------
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
        Sw : float
            Wetted surface (m2)
        """
        super().__init__(cog, mass, inertia, Sw)

        # Velocities
        # FIXME: Vectorial velocities should be considered to can model other
        # aircraft types, like helicopters
        self.TAS = 0  # True Air Speed.
        self.CAS = 0  # Calibrated Air Speed.
        self.EAS = 0  # Equivalent Air Speed.
        self.Mach = 0  # Mach number
        self.aero_vel = np.zeros(3, dtype=np.float)

        # Angular velocities
        self.p = 0  # rad/s
        self.q = 0  # rad/s
        self.r = 0  # rad/s

        # Angles
        self.alpha = 0  # rad
        self.beta = 0  # rad

        # Rate of change of the angle of attack. The angle of attack can be
        # computed as alpha = theta + atan(Uz / Ux), where theta is the pitch
        # angle. Therefore, the rate of change can be computed as
        # alpha_dot = p + (Ux * Uz_dot - Uz * Ux_dot) / (Ux^2 + Uz^2)
        # where p is theta_dot
        self.alpha_dot = 0  # rad/s
        self.beta_dot = 0  # rad/s

        # Provided an "uninitialized" q_inf variable in order to allow the user
        # to set it, instead of using evironment to compute that
        self.__q_inf = None

    @q_inf.setter
    def q_inf(self, q_inf):
        """Impose the considered dynamic pressure at infinity (Pa)

        Parameters
        ----------
        q_inf : float
            Considered dynamic pressure at infinity (Pa)
        """
        self.__q_inf = q_inf

    @property
    def q_inf(self):
        """Get the considered dynamic pressure at infinity (Pa),
        :math:`\\frac{1}{2} \\rho \\TAS^2`

        This method is required by `pyfme.simulator`, so it is preserved for
        backward compatibility

        Returns
        -------
        q_inf : float
            Considered dynamic pressure at infinity (Pa)
        """
        if self.__q_inf is not None:
            return self.__q_inf
        return 0.5 * environment.rho * self.TAS**2

    def update(self, controls, system, environment):
        """Update the aircraft data using the values computed by the simulator

        Parameters
        ----------
        controls : dictionary
            Dictionary with the control values. Outdated data, it is preserved
            just for backward compatibility.
        system : System
            Current simulator system
        environment : Enviroment
            Current Environment settings
        """
        self.aero_vel = system.vel_body - environment.body_wind

        self.alpha, self.beta, self.TAS = calculate_alpha_beta_TAS(
            u=aero_vel[0], v=aero_vel[1], w=aero_vel[2])

        self.p, self.q, self.r = system.vel_ang

        # Setting velocities & dynamic pressure
        self.CAS = tas2cas(self.TAS, environment.p, environment.rho)
        self.EAS = tas2eas(self.TAS, environment.rho)
        self.Mach = self.TAS / environment.a

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
        f, m = super().calculate_forces_and_moments()

        # Add the gravity, which is applied in the global center of gravity
        ff = environment.gravity_vector * self.mass()
        r = self.cog(use_subcomponents=False) - c.cog()
        mm = np.cross(r, ff)

        return f + ff, m + mm
