"""
Module for unit conversion routines. Currently uses the Pint unit conversion
library (https://pint.readthedocs.org) to do the conversions.

Any new units and constants can be added to the text files "units.txt" and
"constants.txt".

NOTE: this is taken from python-common in nomad-lab-base.
It is copied here to remove the dependency from nomad-lab-base.
For more info on python-common visit:
https://gitlab.mpcdf.mpg.de/nomad-lab/python-common

The author of this code is: Dr. Fawzi Roberto Mohamed
E-mail: mohamed@fhi-berlin.mpg.de

"""
from __future__ import print_function
from builtins import str
from builtins import object
import os
import re
import logging
from pint import UnitRegistry
logger = logging.getLogger(__name__)
# disable warnings from pint
logging.getLogger("pint").setLevel(logging.ERROR)
ureg = UnitRegistry(os.path.join(os.path.dirname(__file__), "units.txt"))


def register_userdefined_quantity(quantity, units, value=1):
    """Registers a user defined quantity, valid until redefined.
    The value should be equal to value using units, with value defaulting to 1
    """
    if not re.match(r"usr[A-Za-z]*", quantity):
        raise Exception("User defined quantities should start with usr, and contain only letters")
    if "=" in units:
        raise Exception("Units should not contain an = sign")
    float(value)  # value must be a float number
    ureg.define(quantity + ' = ' + str(value) + " * " + units)


def convert_unit(value, unit, target_unit=None):
    """Converts the given value from the given units to the target units. For
    examples see the bottom section.

    Args:
        value: The numeric value to be converted. Accepts integers, floats,
            lists and numpy arrays
        unit: The units that the value is currently given in as a string. All
            units that have a corresponding declaration in the "units.txt" file
            and combinations like "meter*second**-2" are supported.
        target_unit: The target unit as string. Same rules as for the unit
            argument. If this argument is not given, SI units are assumed.

    Returns:
        The given value in the target units. returned as the same data type as
        the original values.

    .. codeauthor:: Fawzi Mohamed <mohamed@fhi-berlin.mpg.de>

    """

    # Check that the unit is valid
    unit_def = ureg(unit)
    if not unit_def:
        logger.error("Undefined unit given. Cannot do the conversion")
        return

    # If no target is specified, assume SI automatically
    if not target_unit:
        Q_ = ureg.Quantity
        pint_value = Q_(value, unit_def)
        # Base units are defined in the "units.txt" file, and they are the SI units.
        converted_value = pint_value.to_base_units()
        return converted_value.magnitude
    else:
        # Check that the given target unit is valid
        target_unit_def = ureg(target_unit)
        if not target_unit_def:
            logger.error("Undefined target unit given. Cannot do the conversion")
            return
        Q_ = ureg.Quantity
        pint_value = Q_(value, unit_def)
        converted_value = pint_value.to(target_unit_def)
        return converted_value.magnitude


def convert_unit_function_immediate(unit, target_unit=None):
    """Returns a function that converts scalar floats from unit to target_unit
    All units need to be already known.

    For more details see the convert_unit function.
    Could be optimized a bit caching the pint quantities

    Args:
        unit: The units that the value is currently given in as a string. All
            units that have a corresponding declaration in the "units.txt" file
            and combinations like "meter*second**-2" are supported.
        target_unit: The target unit as string. Same rules as for the unit
            argument. If this argument is not given, SI units are assumed.

    Returns:
        The given value in the target units. returned as the same data type as
        the original values.

    .. codeauthor:: Fawzi Mohamed <mohamed@fhi-berlin.mpg.de>

    """
    # Check that the dimensionality of the source and target units match.
    if target_unit is not None:
        source = ureg(target_unit)
        source_dim = source.dimensionality
        target = ureg(unit)
        target_dim = target.dimensionality
        if source_dim != target_dim:
            raise Exception(
                "The dimensionality of unit '{}' does not match the dimensionality of unit '{}'. Cannot do the unit conversion.".format(
                    unit,
                    target_unit))

    return lambda x: convert_unit(x, unit, target_unit)


class LazyF(object):
    """helper class for lazy evaluation of conversion function"""

    def __init__(self, unit, target_unit):
        self.unit = unit
        self.target_unit = target_unit
        self.f = None

    def __call__(self, x):
        if self.f is not None:
            return self.f(x)
        else:
            self.f = convert_unit_function_immediate(self.unit, self.target_unit)
            return self.f(x)


def convert_unit_function(unit, target_unit=None):
    """Returns a function that converts scalar floats from unit to target_unit
    if any of the unit are user defined (usr*), then the conversion is done lazily
    at the first call (i.e. user defined conversions might be undefined when
    calling this)

    For more details see the convert_unit function.
    Could be optimized a bit caching the pint quantities

    Args:
        unit: The units that the value is currently given in as a string. All
            units that have a corresponding declaration in the "units.txt" file
            and combinations like "meter*second**-2" are supported.
        target_unit: The target unit as string. Same rules as for the unit
            argument. If this argument is not given, SI units are assumed.

    Returns:
        The given value in the target units. returned as the same data type as
        the original values.

    .. codeauthor:: Fawzi Mohamed <mohamed@fhi-berlin.mpg.de>

    """
    if "usr" in unit:
        return LazyF(unit, target_unit)
    else:
        return convert_unit_function_immediate(unit, target_unit)


# Testing
if __name__ == "__main__":

    import numpy as np

    # Float
    a = 20
    unit = "angstrom"
    target = "m"
    a_c = convert_unit(a, unit, target)
    print(a_c)

    # Numpy arrays
    b = np.ones((3, 3, 3))
    unit = "angstrom"
    target = "meter"
    b_c = convert_unit(b, unit, target)
    print(b_c)

    # Operators
    c = 20
    unit = "eV*angstrom**-1"
    c_c = convert_unit(c, unit)
    print(c_c)

    # Lists
    d = [20, 10, 0]
    unit = "angstrom"
    target = "m"
    d_c = convert_unit(d, unit, target)
    print(d_c)

    # Temperature
    e = 25
    unit = "celsius"
    target = "kelvin"
    e_c = convert_unit(e, unit, target)
    print(e_c)
