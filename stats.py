"""
Some basic functionality related to statistics, including such things as
the mean, median, mode, midrange, percentile, standard deviation, and
variance. Also some stuff related to distributions.

The idea is to use as little as possible builtin or third party functionality
(except for plotting), hence the implementations are not efficient or good in
any way. This stuff is only useful for learning purposes.

The sample data used is the  the "Old faithful" dataset, which is available
through https://www.aptech.com/blog/the-fundamentals-of-kernel-density-estimation/
for example. Also some data of the electricity price in Finland is used. This can
be found from https://sahko.tk/ for example.
"""

import math
import misc
import kernels
import matplotlib.pyplot as plt

from typing import Optional, Callable


def _read_single_column(fname: str, column: int, delim: str = ',') -> list[float]:
    """
    :param fname:
    :param column:
    :param delim:
    :return:
    """
    data = []
    with open(fname, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(delim)
            data.append(float(line[column]))

    return data


def read_csv(fname: str, columns: int | tuple, delim: str = ',') \
        -> list[float] | list[list[float]]:
    """
    Reads the data from specifically a csv-file
    :param fname: Name of the csv-file
    :param columns:
    :param delim:
    :return:
    """
    if not fname.endswith('.csv'):
        raise ValueError('File must be a csv-file')
    if isinstance(columns, int):
        return _read_single_column(fname=fname, column=columns, delim=delim)
    data = []
    for col in columns:
        data.append(_read_single_column(fname=fname, column=col, delim=delim))

    return data


def mean(data: list[int | float]) -> int | float:
    """
    The arithmetic mean of the given data. This could naturally be
    very easily done with builtin functionality but let us not use them.
    :param data: List of numerical values
    :return:
    """
    summa = 0
    for num in data:
        summa += num
    return summa / len(data)


def median(data: list[int | float]) -> int | float:
    """
    :param data: List of numerical values
    :return:
    """
    data = misc.quicksort(array=data)
    n = len(data)
    if n % 2 != 0:
        return data[n // 2]
    else:
        n2 = n // 2
        return (data[n2 - 1] + data[n2]) / 2


def _minmax(nums: list[int | float]) -> tuple[int | float, int | float]:
    """
    Returns the minimum and maximum value of the given list of numbers.
    Specifically as a tuple of form (minimum, maximum)
    :param nums: List of numerical values
    :return:
    """
    mini, maxi = float('inf'), float('-inf')
    for num in nums:
        if num < mini:
            mini = num
        if num > maxi:
            maxi = num

    return mini, maxi


def midrange(data: list[int | float]) -> int | float:
    """
    :param data: List of numerical values
    :return:
    """
    # Find the minimum and maximum values in the data
    mini, maxi = _minmax(nums=data)
    return (mini + maxi) / 2


def _realmode(data: list[int | float]) -> Optional[list]:
    """
    :param data:
    :return:
    """
    # Find the amount of times the different numbers in the data exist
    counts = {}
    for num in data:
        if num in counts.keys():
            counts[num] += 1
        else:
            counts[num] = 1

    # Find the maximum amount of time any number(s) are present
    maxi = float('-inf')
    for val in counts.values():
        if val > maxi:
            maxi = val
    # If the maximum is one (all numbers are present exactly one
    # time), return None
    if maxi == 1:
        return None
    # Return the number(s) that is/are present the maximum number of times
    return [num for num, times in counts.items() if times == maxi]


def _linspace(start: int | float, end: int | float, n: int) -> list[int | float]:
    """
    :param start:
    :param end:
    :param n:
    :return:
    """
    dx = (end - start) / n
    return [start + dx * i for i in range(n + 1)]


def _hist(data: list[int | float], n: int) -> tuple[list, list]:
    """
    :param data:
    :param n:
    :return:
    """
    mini, maxi = _minmax(nums=data)
    intvals = _linspace(start=mini, end=maxi, n=n)
    fractions = []
    for i in range(1, len(intvals)):
        count = 0
        for num in data:
            if intvals[i - 1] <= num < intvals[i]:
                count += 1

        fractions.append(count)
    return intvals, fractions


def _argmax(nums: list[int | float]) -> int:
    """
    Returns the index of the maximum value in the given list of numbers
    :param nums:
    :return:
    """
    ind, maxi = 0, float('-inf')
    for i, num in enumerate(nums):
        if num > maxi:
            ind = i
            maxi = num

    return ind


def mode(data: list[int | float], n: int = 100) -> list[int | float]:
    """
    Returns the mode(s) of the data. If the data is (more or less) random
    floats (when two or more of the same values will likely exists), divides
    the data into n intervals and returns the midpoint of the interval with
    the largest number of values. In essence, the point where the histogram
    of the data is the highest.
    :param data: List of numerical values
    :param n: Number of intervals to split the divide the data into, in case
    no 'real' mode is found. Defaults to 100.
    :return:
    """
    # Try to find the "real" mode
    real = _realmode(data=data)
    if real is not None:
        return real
    intvals, hist = _hist(data=data, n=n)
    ind = _argmax(hist)
    return [(intvals[ind] + intvals[ind + 1]) / 2]


def percentile_mark(data: list[int | float], num: int | float) -> int | float:
    """
    Returns the percentile mark of the given value, i.e., the percentage of
    data values that are smaller than the given value.
    :param data:
    :param num:
    :return:
    """
    data = misc.quicksort(array=data)
    # The simple cases
    if num > data[-1]:
        return 100
    if num < data[0]:
        return 0
    ind = misc.binary_search(nums=data, value=num)
    return ind / len(data) * 100


def kth_percentile(data: list[int | float], k: int | float) -> int | float:
    """
    Returns the value in the given data that is larger than k per cent of
    the values below it.
    :param data:
    :param k:
    :return:
    """
    k /= 100
    if not 0 <= k <= 1:
        msg = 'The value of k must be in the range 0 ... 100 %'
        raise ValueError(msg)
    n = len(data)
    data = misc.quicksort(array=data)
    ind = k * n
    ind = math.ceil(ind)
    # Check if we round up too high (this can happen if k is very close to 100)
    if ind >= n:
        return data[-1]
    return data[ind]


def variance(data: list[int | float]) -> int | float:
    """
    :param data:
    :return:
    """
    avg = mean(data=data)
    summa = 0
    for x in data:
        diff = x - avg
        summa += diff * diff
    return summa / len(data)


def std(data: list[int | float]) -> int | float:
    """
    :param data:
    :return:
    """
    return math.sqrt(variance(data=data))


def histogram(data: list[int | float], n: int = 20, color: str = 'b',
              fill: bool = True, alpha: float = 1.) -> None:
    """
    Plots a histogram of the data
    :param data:
    :param n:
    :param color:
    :param fill:
    :param alpha:
    :return:
    """
    intvals, fracs = _hist(data=data, n=n)
    _, axis = plt.subplots()
    plt.xlim(intvals[0], intvals[-1])
    _, maxi = _minmax(nums=fracs)
    y_max = maxi * 1.1
    plt.ylim(0, y_max)
    for i in range(len(fracs)):
        # Left side vertical line
        plt.plot([intvals[i], intvals[i]], [0, fracs[i]], c=color)
        # Horizontal line
        plt.plot([intvals[i], intvals[i + 1]], [fracs[i], fracs[i]], c=color)
        # Right side vertical line
        plt.plot([intvals[i + 1], intvals[i + 1]], [0, fracs[i]], c=color)
        if fill:
            y_lim = fracs[i] / y_max
            plt.axvspan(xmin=intvals[i], xmax=intvals[i + 1], ymin=0, ymax=y_lim,
                        alpha=alpha, color=color)


def _kde(kernel: Callable, x: int | float, support: list[int | float],
         h: int | float = None) -> list[int | float]:
    """
    Kernel density estimation:
    https://en.wikipedia.org/wiki/Kernel_density_estimation
    :param kernel:
    :param x:
    :param support:
    :param h:
    :return:
    """
    term = [(v - x) / h for v in support]
    return [kernel(v) for v in term]


def kde(data: list[int | float], kernel: Callable, h: int | float = None,
        width: int = 20) -> None:
    """
    :param data:
    :param kernel:
    :param h:
    :param width:
    :return:
    """
    if h is None:
        h = 1.06 * std(data=data) * math.pow(len(data), -.2)
    div = 1 / (len(data) * h)
    w2 = width // 2
    kerns = []
    # Form the kernel function at each data point
    for point in data:
        support = _linspace(start=point - w2, end=point + w2, n=20)
        kern = _kde(kernel=kernel, x=point, support=support, h=h)
        kerns.append([support, kern])
    # Sum the value of each function at each x-position
    vals = {}
    for xs, ys in kerns:
        for x, y in zip(xs, ys):
            if x in vals.keys():
                vals[x] += y
            else:
                vals[x] = y
    # Sort the dictionary by x
    vals = sorted(vals.items(), key=lambda p: p[0])
    # Put the values back in to a dictionary and scale the y-values
    vals = {k: div * v for k, v in vals}
    # Plot the end result
    plt.plot(vals.keys(), vals.values())
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid()


def main() -> None:
    prices = read_csv(fname='data/hinta.csv', columns=1, delim=';')
    faithful = read_csv(fname='data/old_faithful.csv', columns=2, delim=',')
    kde(data=faithful, kernel=kernels.silverman, width=20, h=None)
    histogram(data=prices, alpha=.2)

    plt.show()


if __name__ == '__main__':
    main()
