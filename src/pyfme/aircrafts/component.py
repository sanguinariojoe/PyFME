"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.
"""

import numpy as np
from fnmatch import fnmatch


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
            of the parent element- of the center of gravity of this element
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
        parent center of gravity

        Returns
        -------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of the parent element- of the center of gravity of this element
            (m, m, m)
        """
        return self.__cog

    @cog.setter
    def cog(self, cog):
        """Set the local center of gravity of the component, refered to the
        parent center of gravity

        Parameters
        ----------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of the parent element- of the center of gravity of this element
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


class Controller(object):
    """Simple controller object
    """
    def __init__(self, name, min_value, max_value, value=0.0):
        """Create the controller

        Parameters
        ----------
        name : string
            A name provided for convenience. Later the controllers can be
            obtained from the aircraft by name, using wildcards.
        min_value : float
            Minimum accepted value. The controller value will be clamped between
            this value and the maximum one
        max_value : float
            Maximum accepted value. The controller value will be clamped between
            this value and the minimum one
        value : float
            Initial value. It will be conveniently clamped
        """
        self.__name = name
        self.__min = min_value
        self.__max = max_value
        self.__value = min(max_value, max(min_value, value))
        self.__callbacks = []
        self.__callback_args = []

    @property
    def name(self):
        """Get the controller' name

        Returns
        -------
        name : string
            A name provided for convenience. Later the controllers can be
            obtained from the aircraft by name, using wildcards.
        """
        return self.__name

    @name.setter
    def name(self, name):
        """Set the name

        Parameters
        ----------
        name : string
            A name provided for convenience. Later the controllers can be
            obtained from the aircraft by name, using wildcards.
        """
        self.__name = name

    @property
    def min_value(self):
        """Get the minimum accepted value

        Returns
        -------
        min_value : float
            Minimum accepted value. The controller value will be clamped between
            this value and the maximum one
        """
        return self.__min

    @min_value.setter
    def min_value(self, min_value):
        """Set the minimum accepted value. All the registered callback functions
        will be called

        Parameters
        ----------
        min_value : float
            Minimum accepted value. The controller value will be clamped between
            this value and the maximum one
        """
        self.__min = min_value
        self.value = max(self.__min, self.value)

    @property
    def max_value(self):
        """Get the minimum accepted value

        Returns
        -------
        max_value : float
            Maximum accepted value. The controller value will be clamped between
            this value and the minimum one
        """
        return self.__max

    @max_value.setter
    def max_value(self, max_value):
        """Set the maximum accepted value. All the registered callback functions
        will be called

        Parameters
        ----------
        max_value : float
            Maximum accepted value. The controller value will be clamped between
            this value and the minimum one
        """
        self.__max = max_value
        self.value = min(self.__max, self.value)

    @property
    def value(self):
        """Get the current controller' value

        Returns
        -------
        value : float
            Current value
        """
        return self.__value

    @value.setter
    def value(self, value):
        """Set the current controller' value. The value will be clamped between
        between the minimum and maximum values. All the registered callback
        functions will be called

        Parameters
        ----------
        value : float
            Current value
        """
        self.__value = min(self.__max, max(self.__min, value))
        for i,c in enumerate(self.__callbacks):
            c(self.__value, self.__callback_args[i])

    def register_callback(self, f, args):
        """Register a callback function to be called when a controller is set.
        This can be useful to control a external linked library, e.g. a graphic
        engine.
        The callback function will be called as follows:
        ``f(self.value, args)``

        Parameters
        ----------
        f : function
            Callback function to be called when value is modified
        args : ...
            Additional arguments for the callback function
        """
        self.__callbacks.append(f)
        self.__callback_args.append(args)


class Component(Structure):
    """Abstract class to define all the possible components of an aircraft,
    including the aircraft itself. The components are structured in a tree,
    model, such that each component may have additional subcomponents.

    The aircraft is characterised by the absence of a parent component
    """
    def __init__(self, cog, mass, inertia, Sw=0.0, parent=None):
        """Create a new component

        Parameters
        ----------
        cog : array_like
            Local x, y, z coordinates -i.e. referered to the center of gravity
            of parent- of the center of gravity of this element (m, m, m)
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
            Parent component which owns the current component
        """
        super().__init__(cog, mass, inertia, Sw)
        self.__parent = parent
        self.__components = []
        self.__controller = None

    @property
    def parent(self):
        """Get the parent component

        Returns
        -------
        parent : Component
            Parent component which owns the current component
        """
        return self.__parent

    @parent.setter
    def parent(self, parent):
        """Set the parent component

        Parameters
        ----------
        parent : Component
            Parent component which owns the current component
        """
        self.__parent = parent

    def top_node(self):
        """Get the parent component

        Returns
        -------
        parent : Component
            Parent component which owns the current component
        """
        top = self
        while top.parent is not None:
            top = top.parent
        return top

    @property
    def components(self):
        """Get the parent component

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
        self.__components = parent

    @property
    def controller(self):
        """Get the controller associated to this component.

        Returns
        -------
        controller : Controller
            Controller. None if a controller has not been associated
        """
        return self.__controller

    @controller.setter
    def controller(self, controller):
        """Set the parent component. Usually the user don't want to manually
        call this function, just because the components which requires a
        controller are automatically registering it

        Parameters
        ----------
        controller : Controller
            Controller associated to this component
        """
        self.__controller = controller

    def get_controllers(wildcard='*'):
        """Get all the controllers of this component and children, with a name
        matching the wildcard.

        Parameters
        ----------
        wildcard : string
            Name wildcard

        Returns
        -------
        controllers : List
            The list of matching controllers
        """
        controllers = []
        for c in self.components:
            controllers += c.get_controllers(wildcard)
        if self.controller is not None and \
           fnmatch(self.controller.name, wildcard):
            controllers.append(self.controller)
        return controllers

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
            should be computed, None if the element COG should be used, i.e.
            ``self.cog(use_subcomponents=True)``
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
        parent, all the required data can be recursively extracted

        Nevertheless this method mis in principle abstract, it is usefull to
        collect the info of the subcomponents

        Returns
        -------
        f : array_like
            Forces integrated from all the subcomponents [N, N, N]
        m : array_like
            Moments integrated from all the subcomponents [N, N, N], respect to
            cog, i.e. ``self.cog(use_subcomponents=False)``
        """
        f = np.zeros(3, dtype=np.float)
        m = np.zeros(3, dtype=np.float)
        cog = self.cog(use_subcomponents=False)
        for c in self.components:
            ff, mm = c.calculate_forces_and_moments()
            # Displace the moment to the new application point
            r = cog - c.cog(use_subcomponents=False)
            mm += np.cross(r, ff)
            # And integrate the forces
            f += ff
            m += mm
        return f, m