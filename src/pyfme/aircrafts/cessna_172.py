# -*- coding: utf-8 -*-
"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline

from pyfme.models.constants import ft2m, slugft2_2_kgm2, lbs2kg
from pyfme.utils.coordinates import wind2body
from pyfme.aircrafts.components import Aircraft, Propeller, Wing, Flap


class Cessna172(Aircraft):
    """
    Cessna 172

    The Cessna 172 aircraft is build using the force and moment coefficients
    described in [1] and [2]. Therefore, since all the coefficients are provided
    considering the entire aircraft, the easiest way to build it up is:
    1.- Create the aircraft with the known mass and inertia
    2.- Add it the known propeller
    3.- Add a single wing, which is representing the wings and the stabilizers
    4.- Register all the known coeffs to the wing
    5.- Create the elevators, alerions and rudder, all of them childs of the
        wing.
    6.- Register their coeffcients

    References
    ----------
    [1] ROSKAM, J., Methods for Estimating Stability and Control
        Derivatives of Conventional Subsonic Airplanes
    [2] McDonnell Douglas Co., The USAF and Stability and Control
        Digital DATCOM, Users Manual
    """

    def __init__(self):
        # We set directly in the aircraft the total mass and inertia. The wetted
        # surface is relegated to the wing
        super().__init__(mass=2300*lbs2kg,
                         inertia=np.diag([948, 1346, 1967])*slugft2_2_kgm2)

        # Setup the propeller
        prop_radius = 0.94  # m
        omega = [1000.0, 2800.0]  # RPM
        J = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94]
        Ct = [0.102122, 0.11097, 0.107621, 0.105191, 0.102446, 0.09947, 0.096775, 0.094706, 0.092341, 0.088912, 0.083878, 0.076336, 0.066669, 0.056342, 0.045688, 0.034716, 0.032492, 0.030253, 0.028001, 0.025735, 0.023453, 0.021159, 0.018852, 0.016529, 0.014194, 0.011843, 0.009479, 0.0071, 0.004686, 0.002278, -0.0002, -0.002638, -0.005145, -0.007641, -0.010188]
        self.components.append(Propeller(parent=self,
                                         r=prop_radius,
                                         omega=omega,
                                         J=J,
                                         Ct=Ct))

        # Setup the wings and stabilizers in a single wing
        chord = 1.49352  # m
        span = 10.91184  # m
        Sw = 16.2        # m2
        wing = Wing(chord, span, Sw, self)
        self.components.append(wing)

        # Register the coefficients of the wing
        alpha = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 17, 18, 19.5]  # deg
        # Lift, lateral and drag forces due to the attack angle (see [2]). The
        # lateral force is also multiplied by the sideslip angle, beta.
        CL = [-0.571, -0.321, -0.083, 0.148, 0.392, 0.65, 0.918, 1.195, 1.659, 1.789, 1.84, 1.889]
        CY = [-0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268, -0.268]
        CD = [0.044, 0.034, 0.03, 0.03, 0.036, 0.048, 0.067, 0.093, 0.15, 0.169, 0.177, 0.184]
        wing.add_force_coeff(alpha, CL, "CL(alpha)")
        wing.add_force_coeff(alpha, CY, "CY(alpha)")
        wing.add_force_coeff(alpha, CD, "CD(alpha)")
        # Moments due to the attack angle (see [2]). Both Cl and Cn are
        # multiplied by the sideslip angle, beta.
        # XXX: For some reason, Cl is multiplied by 0.1 at the time of getting
        # the forces in the old implementation
        Cl = 0.1 * np.array([-0.178, -0.186, -0.1943, -0.202, -0.2103, -0.219, -0.2283, -0.2376, -0.2516, -0.255, -0.256, -0.257])
        Cm = [0.0597, 0.0498, 0.0314, 0.0075, -0.0248, -0.068, -0.1227, -0.1927, -0.3779, -0.4605, -0.5043, -0.5496]
        Cn = [0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126, 0.0126]
        wing.add_force_coeff(alpha, Cl, "Cl(alpha)")
        wing.add_force_coeff(alpha, Cm, "Cm(alpha)")
        wing.add_force_coeff(alpha, Cn, "Cn(alpha)")
        # Lift force and pitch moment due to the attack angle, and its rate of
        # change (see [2]).
        CL_alphadot = [2.434, 2.362, 2.253, 2.209, 2.178, 2.149, 2.069, 1.855, 1.185, 0.8333, 0.6394, 0.4971]
        Cm_alphadot = [-6.64, -6.441, -6.146, -6.025, -5.942, -5.861, -5.644, -5.059, -3.233, -2.273, -1.744, -1.356]
        wing.add_force_coeff(alpha, CL_alphadot, "CL(alpha, alphadot)")
        wing.add_force_coeff(alpha, Cm_alphadot, "Cm(alpha, alphadot)")
        # Lift force and pitch moment due to the attack angle, and pitch
        # rotation (see [2]).
        CL_q = [7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282, 7.282]
        Cm_q = [-6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232, -6.232]
        # FIXME: CM_q multiplicado por 2 hasta que alpha_dot pueda ser calculado
        Cm_q *= 2
        wing.add_force_coeff(alpha, CL_q, "CL(alpha, q)")
        wing.add_force_coeff(alpha, Cm_q, "Cm(alpha, q)")
        # Lateral force, roll moment and yaw moment due to the attack angle, and
        # roll rotation (see [2]).
        CY_p = [-0.032, -0.0372, -0.0418, -0.0463, -0.051, -0.0563, -0.0617, -0.068, -0.0783, -0.0812, -0.0824, -0.083]
        Cl_p = [-0.4968, -0.4678, -0.4489, -0.4595, 0.487, -0.5085, -0.5231, -0.4916, -0.301, -0.203, -0.1498, -0.0671]
        Cn_p = [0.03, 0.016, 0.00262, -0.0108, -0.0245, -0.0385, -0.0528, -0.0708, -0.113, -0.1284, -0.1356, -0.1422]
        wing.add_force_coeff(alpha, CY_p, "CY(alpha, p)")
        wing.add_force_coeff(alpha, Cl_p, "Cl(alpha, p)")
        wing.add_force_coeff(alpha, Cn_p, "Cn(alpha, p)")
        # Lateral force, roll moment and yaw moment due to the attack angle, and
        # yaw rotation (see [2]).
        CY_r = [0.2018, 0.2054, 0.2087, 0.2115, 0.2139, 0.2159, 0.2175, 0.2187, 0.2198, 0.2198, 0.2196, 0.2194]
        Cl_r = [-0.09675, -0.05245, -0.01087, 0.02986, 0.07342, 0.1193, 0.1667, 0.2152, 0.2909, 0.3086, 0.3146, 0.3197]
        Cn_r = [-0.028, -0.027, -0.027, -0.0275, -0.0293, -0.0325, -0.037, -0.043, -0.05484, -0.058, -0.0592, -0.06015]
        wing.add_force_coeff(alpha, CY_r, "CY(alpha, r)")
        wing.add_force_coeff(alpha, Cl_r, "Cl(alpha, r)")
        wing.add_force_coeff(alpha, Cn_r, "Cn(alpha, r)")
        
        # Create the elevators as a single component inside the wing
        elevator = Flap(wing, [-26, -20, -10, -5, 0, 7.5, 15, 22.5, 28],
                        controller_name='delta_elevator')
        wing.components.append(elevator)
        # Lift force, drag force and pitch moment (see [2])
        # Actually the Lift force and pitch moment does not depends on the angle
        # of attack, but for the sake of simplicity we are implicetly adding
        # such dependency
        CL = np.transpose([[-0.132, -0.123, -0.082, -0.041, 0, 0.061, 0.116, 0.124, 0.137]] * len(alpha))
        CD = [[0.0135, 0.0119, 0.0102, 0.00846, 0.0067, 0.0049, 0.00309, 0.00117, -0.0033, -0.00541, -0.00656, -0.00838],
              [0.0121, 0.0106, 0.00902, 0.00738, 0.00574, 0.00406, 0.00238, 0.00059, -0.00358, -0.00555, -0.00661, -0.00831],
              [0.00651, 0.00552, 0.00447, 0.00338, 0.00229, 0.00117, 0.0000517, -0.00114, -0.00391, -0.00522, -0.00593, -0.00706],
              [0.00249, 0.002, 0.00147, 0.000931, 0.000384, -0.000174, -0.000735, -0.00133, -0.00272, -0.00337, -0.00373, -0.00429],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [-0.00089, -0.00015, 0.00064, 0.00146, 0.00228, 0.00311, 0.00395, 0.00485, 0.00693, 0.00791, 0.00844, 0.00929],
              [0.00121, 0.00261, 0.00411, 0.00566, 0.00721, 0.00879, 0.0104, 0.0121, 0.016, 0.0179, 0.0189, 0.0205],
              [0.00174, 0.00323, 0.00483, 0.00648, 0.00814, 0.00983, 0.0115, 0.0133, 0.0175, 0.0195, 0.0206, 0.0223],
              [0.00273, 0.00438, 0.00614, 0.00796, 0.0098, 0.0117, 0.0135, 0.0155, 0.0202, 0.0224, 0.0236, 0.0255]]
        Cm = np.transpose([[0.3302, 0.3065, 0.2014, 0.1007, -0.0002, -0.1511, -0.2863, -0.3109, -0.345]] * len(alpha))
        elevator.add_force_coeff(alpha, CL, "CL(alpha)")
        elevator.add_force_coeff(alpha, CD, "CD(alpha)")
        elevator.add_force_coeff(alpha, Cm, "Cm(alpha)")

        # Create the ailerons as another component inside the wing
        aileron = Flap(wing, [-15, -10, -5, -2.5, 0, 5, 10, 15, 20],
                       controller_name='delta_aileron')
        wing.components.append(aileron)
        # Roll and yaw moments (see [2])
        # Again, the roll moment is not affected by the angle of attack, but
        # but it is simpler to implicetly add such dependency
        Cl = np.transpose([[-0.078052, -0.059926, -0.036422, -0.018211, 0, 0.018211, 0.036422, 0.059926, 0.078052]] * len(alpha))
        Cn = [[-0.004321, -0.002238, -0.0002783, 0.001645, 0.003699, 0.005861, 0.008099, 0.01038, 0.01397, 0.01483, 0.01512, 0.01539],
              [-0.003318, -0.001718, -0.0002137, 0.001263, 0.00284, 0.0045, 0.006218, 0.00797, 0.01072, 0.01138, 0.01161, 0.01181],
              [-0.002016, -0.001044, -0.000123, 0.0007675, 0.00173, 0.002735, 0.0038, 0.004844, 0.00652, 0.00692, 0.00706, 0.0072],
              [-0.00101, -0.000522, -0.0000649, 0.000384, 0.000863, 0.00137, 0.0019, 0.00242, 0.00326, 0.00346, 0.00353, 0.0036],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.00101, 0.00052, 0.000065, -0.000384, -0.00086, -0.0014, -0.002, -0.002422, -0.00326, -0.00346, -0.00353, -0.0036],
              [0.00202, 0.001044, 0.00013, -0.0008, -0.00173, -0.002735, -0.0038, -0.004844, -0.00652, -0.00692, -0.00706, -0.0072],
              [0.00332, 0.00172, 0.000214, -0.001263, -0.00284, -0.0045, -0.00622, -0.008, -0.01072, -0.01138, -0.01161, -0.01181],
              [0.004321, 0.00224, 0.00028, -0.001645, -0.0037, -0.00586, -0.0081, -0.0104, -0.014, -0.01483, -0.01512, -0.0154]]
        aileron.add_force_coeff(alpha, Cl, "Cl(alpha)")
        aileron.add_force_coeff(alpha, Cn, "Cn(alpha)")

        # Finally, create the rudder. Actually the rudder has not the same
        # orientation of the wing. However, since all the coefficients are
        # defined as a whole wing, it is more convenient for us not taking into
        # account such fact.
        rudder_angles = np.linspace(-16.0, 16.0, num=9)
        rudder = Flap(wing, rudder_angles,
                      controller_name='delta_rudder')
        wing.components.append(rudder)
        # Roll and yaw moments (see [1]). This time, the coefficients linearly
        # depends on the rudder angle
        Cl = 0.075 * np.outer(
            [-0.091, -0.082, -0.072, -0.063, -0.053, -0.0432, -0.0333, -0.0233, -0.0033, 0.005, 0.009, 0.015],
            np.deg2rad(rudder_angles))
        Cn = 0.075 * np.outer(
            [0.211, 0.215, 0.218, 0.22, 0.224, 0.226, 0.228, 0.229, 0.23, 0.23, 0.23, 0.23],
            np.deg2rad(rudder_angles))
        rudder.add_force_coeff(alpha, Cl, "Cl(alpha)")
        rudder.add_force_coeff(alpha, Cn, "Cn(alpha)")

        # Aerodynamic Coefficients. They are preserved for backward
        # compatibility
        self.CL, self.CD, self.Cm = 0, 0, 0
        self.CY, self.Cl, self.Cn = 0, 0, 0

        # Thrust Coefficient. preserved for backward compatibility
        self.Ct = 0

        # Force and moments. preserved for backward compatibility
        self.total_forces = np.zeros(3)
        self.total_moments = np.zeros(3)

        # Provide a way to access the controls on the old fashion. Just for
        # backward compatibility, it should be removed in future versions
        self.controls = {'delta_elevator': 0,
                         'delta_aileron': 0,
                         'delta_rudder': 0,
                         'delta_t': 0}
        self.control_limits = {'delta_elevator': (-26, 28),  # deg
                               'delta_aileron': (-15, 20),  # deg
                               'delta_rudder': (-16, 16),  # deg
                               'delta_t': (0, 1)}  # non-dimensional

    def update(self, controls, system, environment):
        """This function is overloaded just for backward compatibility. This is
        just setting the controls updated on the old-fashion way. This
        methodology should become considered deprecated. The new system provides
        a much more flexible and robust way to access and modify controls

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
        for control_name, control_value in controls.items():
            controllers = self.get_controllers(control_name)
            for c in controllers:
                c.value = control_value
                self.controls[control_name] = c.value
        super().update(controls, system, environment)

    def calculate_forces_and_moments(self):
        """This function is overloaded just for backward compatibility. This is
        just calling the base function, and then storing the forces and moments
        as class attributes

        Returns
        -------
        f : array_like
            Drag, lateral and Lift forces (N)
        m : array_like
            Roll, pitch and yaw moments (N * m)
        """
        f, m = super().calculate_forces_and_moments()
        self.total_forces, self.total_moments = f, m

        f_fact = self.q_inf * self.Sw()
        if f_fact == 0.0:
            return f, m
        wing = self.components[-1]
        c = wing.chord
        b = wing.span
        m_fact = np.array([b, c, b], dtype=np.float) * f_fact

        self.CD, self.CY, self.CL = f / f_fact
        self.Cl, self.Cm, self.Cn = m / m_fact

        return f, m
