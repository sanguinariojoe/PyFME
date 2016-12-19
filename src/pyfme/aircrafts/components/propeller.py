"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

from pyfme.environment.environment import Environment
from pyfme.aircrafts import Component, Controller
from pyfme.aircrafts.components import Aircraft
import numpy as np


class Propeller(Component):
    """A propeller. The propeller is generating a thrust depending on the value
    of the internal controller, which is automatically created. Such controller
    may take values between 0 (minimum thrust) and 1 (maximum thrust)
    """
    def __init__(self, r, omega, J, Ct,
                 vec=np.asarray([1, 0, 0]),
                 controller_name='delta_t'
                 cog=np.zeros(3, dtype=np.float),
                 mass=0.0,
                 inertia=np.zeros((3, 3), dtype=np.float),
                 Sw=0.0,
                 parent=None):
        """Create a new propeller

        Parameters
        ----------
        r : float
            Propeller radius (m)
        omega : array_like
            List of considered propeller angular velocities (RPM). The current
            rpm are linearly interpolated using the controller value, which can
            take values between 0 and 1
        J : array_like
            Advance ratio considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``Ct``
        Ct : array_like
            Thrust coeff. considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``J``
        vec : array_like
            Thrust direction vector
        controller_name : string
            Name of the associated controller to be automatically generated
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
        parent : Component
            Parent component which owns the current component.
        """
        super().__init__(cog, mass, inertia, Sw, parent=parent)

        # Velocities
        self.__r = r
        self.__delta_t = np.linspace(0, 1, num=len(omega))
        self.__omega = np.asarray(omega)
        self.__J = J
        self.__Ct = Ct
        self.__vec = vec
        self.controller = Controller(controller_name, 0.0, 1.0)

    @property
    def r(self):
        """Propeller radius (m)

        Returns
        -------
        r : float
            Propeller radius (m)
        """
        return self.__r

    @r.setter
    def r(self, r):
        """Set the propeller radius (m)

        Parameters
        ----------
        r : float
            Propeller radius (m)
        """
        self.__r = r

    @property
    def omega(self):
        """List of considered propeller angular velocities (RPM)

        Returns
        -------
        omega : array_like
            List of considered propeller angular velocities (RPM). The current
            rpm are linearly interpolated using the controller value, which can
            take values between 0 and 1
        """
        return self.__omega

    @omega.setter
    def omega(self, omega):
        """Set the list of considered propeller angular velocities (RPM)

        Parameters
        ----------
        omega : array_like
            List of considered propeller angular velocities (RPM). The current
            rpm are linearly interpolated using the controller value, which can
            take values between 0 and 1
        """
        self.__delta_t = np.linspace(0, 1, num=len(omega))
        self.__omega = np.asarray(omega)

    @property
    def J(self):
        """Advance ratio considered values

        Returns
        -------
        J : array_like
            Advance ratio considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``Ct``
        """
        return self.__omega

    @J.setter
    def J(self, J):
        """Set the advance ratio considered values

        Parameters
        ----------
        J : array_like
            Advance ratio considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``Ct``
        """
        self.__J = J

    @property
    def Ct(self):
        """Thrust coeff. considered values

        Returns
        -------
        Ct : array_like
            Thrust coeff. considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``J``
        """
        return self.__Ct

    @Ct.setter
    def Ct(self, Ct):
        """Set the thrust coeff. considered values

        Parameters
        ----------
        Ct : array_like
            Thrust coeff. considered values. The propeller thrust value will be
            computed getting first the current advance ratio, interpolating
            later the trhust coefficient using this array and ``J``
        """
        self.__Ct = Ct

    @property
    def vec(self):
        """Thrust direction vector

        Returns
        -------
        vec : array_like
            Thrust direction vector
        """
        return self.__vec

    @vec.setter
    def vec(self, Ct):
        """Set the thrust direction vector

        Parameters
        ----------
        vec : array_like
            Thrust direction vector
        """
        self.__vec = vec

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

        # Get the airspeed (just in case we have an available aircraft)
        V = np.zeros(3, dtype=np.float)
        aircraft = self.top_node()
        if isinstance(aircraft, Aircraft):
            # FIXME: Vectorial velocities should be considered to can model
            # other aircraft types, like helicopters
            V[0] = aircraft.TAS
        V = np.dot(V, self.__vec)

        delta_t = self.controller.value
        rho = environment.rho
        omega = np.interp(delta_t, self.__delta_t, self.__omega)  # rpm
        omega_RAD = (omega * 2.0 * np.pi) / 60.0  # rad/s

        J = (np.pi * V) / (omega_RAD * self.__r)
        Ct = np.interp(J, self.__J, self.__Ct)
        T = (2.0 / np.pi)**2 * rho * (omega_RAD * self.__r)**2 * Ct  # N

        ff = T * self.__vec
        r = self.cog(use_subcomponents=False) - self.cog()
        mm = np.cross(r, ff)

        return f + ff, m + mm
