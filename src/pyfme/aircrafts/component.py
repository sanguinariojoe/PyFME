"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

import numpy as np


class Structure(object):
    """Structural element. In PyFME the structural elements are just abstract
    mass elements, with an associated inertia and center of gravity.

    This class is provided just for convenience, because actually all the
    components are structural elements. Along this line, as the Component class
    itself, this class should be considered as purely abstract
    """
    def __init__(self, cog, mass, inertia, Sw=0.0):
        """Create an structural element

        Parameters
        ----------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of the partner element- of the center of gravity of this element
            (m, m, m)
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
        self.__cog = np.asarray(cog)
        self.__mass = mass
        self.__inertia = np.asarray(inertia)
        self.__Sw = Sw


    @property
    def cog(self):
        """Get the local center of gravity of the component, refered to the
        partner center of gravity

        Returns
        -------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of the partner element- of the center of gravity of this element
            (m, m, m)
        """
        return self.__cog

    @cog.setter
    def cog(self, cog):
        """Set the local center of gravity of the component, refered to the
        partner center of gravity

        Parameters
        ----------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of the partner element- of the center of gravity of this element
            (m, m, m)
        """
        self.__cog = np.asarray(cog)

    @property
    def mass(self):
        """Get the mass of the element

        Returns
        -------
        mass : float
            Mass of the component (kg)
        """
        return self.__mass

    @mass.setter
    def mass(self, mass):
        """Set the mass of the element

        Parameters
        ----------
        mass : float
            Mass of the component (kg)
        """
        self.__mass = mass
        
    @property
    def inertia(self):
        """Get the 3x3 inertia tensor

        Returns
        -------
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft
        """
        return self.__inertia

    @inertia.setter
    def inertia(self, inertia):
        """Set the 3x3 inertia tensor

        Parameters
        ----------
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft
            Current equations assume that the global aircraft has a symmetry
            plane (x_b - z_b), thus J_xy and J_yz must be null
        """
        self.__inertia = np.asarray(inertia)

    @property
    def Sw(self):
        """Get the wetted surface of the element

        Returns
        -------
        mass : float
            Wetted surface of the component (kg)
        """
        return self.__Sw

    @Sw.setter
    def Sw(self, Sw):
        """Set the wetted surface of the element

        Parameters
        ----------
        mass : float
            Wetted surface of the component (kg)
        """
        self.__Sw = Sw
        
    def get_inertia(self, o=None):
        """Get the 3x3 inertia tensor respect to an specific point. See
        https://en.wikipedia.org/wiki/Parallel_axis_theorem

        Parameters
        ----------
        o : array_like
            x, y, z coordinates of the point from which the inertia tensor
            should be computed, None if the element COG should be used

        Returns
        -------
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft, repect to o point
        """
        if o is None:
            return self.inertia

        r = o - self.cog
        j = self.mass * (np.dot(r, r) * np.eye(3) + np.outer(r, r))
        return self.inertia + j


class Component(Structure):
    """Abstract class to define all the possible components of an aircraft,
    including the aircraft itself. The components are structured in a tree,
    model, such that each component may have additional subcomponents.

    The aircraft is characterised by the absence of a partner component
    """
    def __init__(self, cog, mass, inertia, Sw=0.0, partner=None):
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
        partner : Component
            Partner component which owns the current component
        """
        super().__init__(cog, mass, inertia, Sw)
        self.__partner = partner
        self.__components = []

    @property
    def partner(self):
        """Get the partner component

        Returns
        -------
        partner : Component
            Partner component which owns the current component
        """
        return self.__partner

    @partner.setter
    def partner(self, partner):
        """Set the partner component

        Parameters
        ----------
        partner : Component
            Partner component which owns the current component
        """
        self.__partner = partner

    @property
    def components(self):
        """Get the partner component

        Returns
        -------
        components : list
            List of components belonging to this one
        """
        return self.__components

    @components.setter
    def components(self, components):
        """Set the list of components belonging to this one.
        In general using this setter is not recommended, becoming much more
        convenient the usage of the getter to append new components

        Parameters
        ----------
        components : list
            List of components belonging to this one
        """
        self.__components = partner

    def cog(self, use_subcomponents=True):
        """Get the cog coordinates, taking into account the mass of the
        subcomponents

        Parameters
        ----------
        use_subcomponents : bool
            Choose whether the subcomponents should be considered or not

        Returns
        -------
        cog : array_like
            Center of gravity, eventually modified by the subcomponents
        """
        if not use_subcomponents:
            return super().cog

        m_cog = np.zeros(3, dtype=np.float)
        for c in self.components:
            m_cog += c.mass() * c.cog()

        m_cog += super().mass * super().cog

        return m_cog / self.mass()

    def mass(self, use_subcomponents=True):
        """Get the component' mass, taking into account the children components

        Parameters
        ----------
        use_subcomponents : bool
            Choose whether the subcomponents should be considered or not

        Returns
        -------
        mass : float
            Mass of the component, eventually including all the subcomponents
        """
        if not use_subcomponents:
            return super().mass

        mass = 0.0
        for c in self.components:
            mass += c.mass()

        mass += super().mass

        return mass

    def inertia(self, o=None, use_subcomponents=True):
        """Get the 3x3 inertia tensor respect to an specific point. See
        https://en.wikipedia.org/wiki/Parallel_axis_theorem
        This method is taking into account the subcomponents

        Parameters
        ----------
        o : array_like
            x, y, z coordinates of the point from which the inertia tensor
            should be computed, None if the element COG should be used
        use_subcomponents : bool
            Choose whether the subcomponents should be considered or not

        Returns
        -------
        inertia : array_like
            3x3 tensor of inertia of the component (kg * m2) for the upright
            aircraft, repect to o point, eventually considering the
            subcomponents
        """
        if not use_subcomponents:
            return super().get_inertia(o)

        # The parallel axis theorem, wrongly known as Steiner's theorem,
        # requires to use the minimum Inertia point -i.e. the COG- as reference
        # one. Otherwise, it cannot be granted the sign of the displacement
        # Inertia moment.
        cog = self.cog()
        inertia = np.zeros((3,3), dtype=np.float)
        for c in self.components:
            inertia += c.inertia(cog)

        inertia += super().get_inertia(cog)

        if o is None:
            # That's all, this is exactly the point we have been considering
            return inertia

        r = o - cog
        inertia += self.mass() * (np.dot(r, r) * np.eye(3) + np.outer(r, r))

        return inertia

    def Sw(self, use_subcomponents=True):
        """Get the component' wetted surface, taking into account the children
        components

        Parameters
        ----------
        use_subcomponents : bool
            Choose whether the subcomponents should be considered or not

        Returns
        -------
        Sw : float
            Wetted surface of the component, eventually including all the
            subcomponents
        """
        if not use_subcomponents:
            return super().Sw

        Sw = 0.0
        for c in self.components:
            Sw += c.Sw()

        Sw += super().Sw

        return Sw

    def calculate_forces_and_moments(self):
        """Abstract method to become overloaded by each specific component.

        In principle, the required parameters may depend on the type of
        component. However, since each component has info regarding who is its
        partner, all the required data can be recursively extracted

        Nevertheless this method mis in principle abstract, it is usefull to
        collect the info of the subcomponents

        Returns
        -------
        f : array_like
            Forces integrated from all the subcomponents [N, N, N]
        m : array_like
            Moments integrated from all the subcomponents [N, N, N], respect to
            cog
        """
        f = np.zeros(3, dtype=np.float)
        m = np.zeros(3, dtype=np.float)
        cog = self.cog()
        for c in self.components:
            ff, mm = c.calculate_forces_and_moments()
            # Displace the moment to the new application point
            r = cog - c.cog()
            mm += np.cross(r, ff)
            # And integrate the forces
            f += ff
            m += mm
        return f, m
