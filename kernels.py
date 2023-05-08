"""
Some common kernel functions are stored here. All the functions can
be found from Wikipedia:
https://en.wikipedia.org/wiki/Kernel_(statistics)#In_non-parametric_statistics
"""

import math
import misc


def boxcar(_: int | float) -> int | float:
    """
    The function does not really need or use the parameter but the function
    definition must include it so that it's signature matches the other
    kernel functions' signature
    :param _:
    :return:
    """
    return .5


def triangular(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return 1 - abs(u)


def epanechnikov(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return 3 / 4 * (1 - u * u)


def quartic(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return 15 / 16 * math.pow(1 - u * u, 2)


def triweight(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return 35 / 32 * math.pow(1 - u * u, 3)


def tricube(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return 70 / 81 * math.pow(1 - math.pow(abs(u), 3), 2)


def gaussian(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    sq = 1 / math.sqrt(2 * math.pi)
    return sq * math.exp(-.5 * u * u)


def cosine(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    u = misc.clamp(val=u, low=-1, high=1)
    return math.pi / 4 * math.cos(math.pi / 2 * u)


def logistic(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    return 1 / (math.exp(u) + 2 + math.exp(-u))


def sigmoid(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    return 2 / math.pi * 1 / (math.exp(u) + math.exp(-u))


def silverman(u: int | float) -> int | float:
    """
    :param u:
    :return:
    """
    ausq = abs(u) / math.sqrt(2)
    return .5 * math.exp(-ausq) * math.sin(ausq + math.pi / 4)
