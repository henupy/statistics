"""
Some basic functionality related to statistics, including such things as
the mean, median, mode, midrange, percentile, standard deviation, and
variance. Also some stuff related to distributions.

The idea is to use implement at least the majority of the things without using
builtin or thirdparty functionality,, hence the implementations are not
efficient or good in any way. This stuff is only useful for learning purposes.

The sample data used is the  the "Old faithful" dataset, which is available
through https://www.aptech.com/blog/the-fundamentals-of-kernel-density-estimation/
for example. Also some data about the electricity price in Finland is used. This
data can be found from https://sahko.tk/ for example.
"""

import math
import misc
import random
import kernels
import functools as ft
import matplotlib.pyplot as plt

from typing import Optional, Callable


def _read_single_column(fname: str, column: int, delim: str = ",") -> list[float]:
    """
    :param fname: Name of the file
    :param column: Which column to read
    :param delim: What character is used to separate the columns
    :return:
    """
    data = []
    with open(fname, "r") as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(delim)
            data.append(float(line[column]))

    return data


def read_csv(fname: str, columns: int | tuple, delim: str = ",") \
        -> list[float] | list[list[float]]:
    """
    Reads the data from specifically a csv-file
    :param fname: Name of the csv-file
    :param columns: The column(s) to read
    :param delim: The character used to split the columns. Can be for
        example ";" or ",".
    :return:
    """
    if not fname.endswith(".csv"):
        raise ValueError("File must be a csv-file")
    if isinstance(columns, int):
        return _read_single_column(fname=fname, column=columns, delim=delim)
    data = []
    for col in columns:
        data.append(_read_single_column(fname=fname, column=col, delim=delim))

    return data


def mean(data: list[int | float]) -> int | float:
    """
    The arithmetic mean of the given data
    :param data: List of numbers
    :return:
    """
    summa = 0
    for num in data:
        summa += num
    return summa / len(data)


def median(data: list[int | float]) -> int | float:
    """
    Returns either the value at the middle of the list after it is sorted
    in ascending order, or the average of the two middle-most values if
    the length of the array is even.
    :param data: List of numbers
    :return:
    """
    data = misc.quicksort(array=data)
    n = len(data)
    if n % 2 != 0:
        return data[n // 2]
    else:
        n2 = n // 2
        return (data[n2 - 1] + data[n2]) / 2


def midrange(data: list[int | float]) -> int | float:
    """
    Returns the average of the smallest and largest number in the array
    :param data: List of numbers
    :return:
    """
    return (min(data) + max(data)) / 2


def _realmode(data: list[int | float]) -> Optional[list]:
    """
    Finds the "real" mode of the given list of numbers, if it exists. "Real"
    mode here means that the mode is actually present in the data.
    :param data: List of numbers
    :return:
    """
    # Find the amount of times the different numbers in the data exist
    counts = {}
    for num in data:
        if num in counts.keys():
            counts[num] += 1
        else:
            counts[num] = 1

    maxi = max(counts.values())
    # If the maximum is one (all numbers are present exactly one
    # time), return None
    if maxi == 1:
        return None
    # Return the number(s) that is/are present the maximum number of times
    return [num for num, times in counts.items() if times == maxi]


def _linspace(start: int | float, end: int | float, n: int) -> list[int | float]:
    """
    Creates a list of n numbers in the range start ... end. The ending value
    is included.
    :param start: Starting point/value
    :param end: Ending point/value
    :param n: The amount of intervals to split the range
    :return:
    """
    dx = (end - start) / n
    return [start + dx * i for i in range(n + 1)]


def _hist(data: list[int | float], n: int) -> tuple[list, list]:
    """
    A convenience function for finding the "fake" mode, which is
    basically a histogram. Is also used in plotting of the histogram.
    :param data: List of numbers
    :param n: The amount of bins/intervals to split the data, i.e.,
        how many bars will the histogram have.
    :return:
    """
    intvals = _linspace(start=min(data), end=max(data), n=n)
    counts = []
    for i in range(1, len(intvals)):
        count = 0
        for num in data:
            if intvals[i - 1] <= num < intvals[i]:
                count += 1

        counts.append(count)
    return intvals, counts


def _argmax(nums: list[int | float]) -> int:
    """
    Returns the index of the maximum value in the given list of numbers
    :param nums: List of numbers
    :return:
    """
    return nums.index(max(nums))


def mode(data: list[int | float], n: int = 100) -> list[int | float]:
    """
    Returns the mode(s) of the data. If any one value does not exist twice
    or more times in the data, divides the data into n intervals and returns
    the midpoint of the interval with the largest number of values. In
    essence, the point where the histogram of the data is the highest.
    :param data: List of numerical values
    :param n: Number of intervals to split the divide the data into, in case
        no "real" mode is found. Defaults to 100.
    :return:
    """
    # Try to find the "real" mode
    real = _realmode(data=data)
    if real is not None:
        return real
    # Calculate the histogram and find the bin with largest amount of values
    intvals, hist = _hist(data=data, n=n)
    ind = _argmax(hist)
    return [(intvals[ind] + intvals[ind + 1]) / 2]


def percentile_value(data: list[int | float], num: int | float) -> int | float:
    """
    Returns the percentile of the given value, i.e., the percentage of
    data values that are smaller than the given value.
    :param data: List of numbers
    :param num: A number that can or can not be present in the data
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
    :param data: List of numbers
    :param k: The percentile, must be in range 0 ... 100
    :return:
    """
    k /= 100
    if not 0 <= k <= 1:
        msg = "The value of k must be in the range 0 ... 100 %"
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
    :param data: List of numbers
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
    :param data: List of numbers
    :return:
    """
    return math.sqrt(variance(data=data))


def histogram(data: list[int | float], n: int = 20, norm: bool = True,
              color: str = "b", fill: bool = True, alpha: float = 1.,
              xlabel: str = "Values", ylabel: str = "Density") -> None:
    """
    Plots a histogram of the data
    :param data: List of numbers
    :param n: The number of bins/intervals to split the data into
    :param norm: Whether to normalise the data or not
    :param color: The color of the histogram
    :param fill: Whether to fill the histogram
    :param alpha: The opaqueness of the filled part of the histogram
    :param xlabel: Label for the x-axis of the plot. Default is "Value"
    :param ylabel: Label for the y-axis of the plot. Default is "Density"
    :return:
    """
    intvals, fracs = _hist(data=data, n=n)
    if norm:
        tot = len(data)
        fracs = [v / tot for v in fracs]
    _, axis = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(intvals[0], intvals[-1])
    y_max = max(fracs) * 1.1
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
         h: int | float) -> list[int | float]:
    """
    A convenience function used in kernel density estimation
    :param kernel: The kernel function to be used (see kernels.py)
    :param x: The point around which the kernel function's values are
        calculated
    :param support: A linear support vector of points around the x point
    :param h: Bandwidth
    :return:
    """
    term = [(v - x) / h for v in support]
    return [kernel(v) for v in term]


def kde(data: list[int | float], kernel: Callable, h: int | float = None,
        width: int | float = 20, dx: int | float = 1,
        xlabel: str = "Values", ylabel: str = "Density") -> None:
    """
    Function to perform kernel density estimation. The idea is to try
    to estimate the (shape of the) probability density function of
    the given data. More info can be found e.g. from Wikipedia:
    https://en.wikipedia.org/wiki/Kernel_density_estimation
    :param data: List of numbers
    :param kernel: The kernel function to be used in the estimation
        (see kernels.py)
    :param h: Bandwidth
    :param width: The width of the sub-region of the data range
        in which the kernel function's values are solved
    :param dx: Width of a single step used in the linear support
        vector
    :param xlabel: Label for the x-axis of the plot. Default is "Value"
    :param ylabel: Label for the y-axis of the plot. Default is "Density"
    :return:
    """
    # Slight error checking regarding the generation of the support vectors
    if width < dx:
        msg = f"The width of a single step can not be smaller than " \
              f"the width of the whole region. Now got {width=} < " \
              f"{dx=}."
        raise ValueError(msg)
    # Form the kernel function at each data point
    if h is None:
        # This correlation for the bandwidth can be found from the Wikipedia
        # article mentioned above
        h = 1.06 * std(data=data) * math.pow(len(data), -.2)
    n = int(width // dx)
    div = 1 / (len(data) * h)
    w2 = width // 2
    kerns = []
    for point in data:
        support = _linspace(start=point - w2, end=point + w2, n=n)
        kern = _kde(kernel=kernel, x=point, support=support, h=h)
        kerns.append([support, kern])
    # Sum the value of each function at each x-position that we have
    # y-values for
    vals = {}
    for xs, ys in kerns:
        for x, y in zip(xs, ys):
            if x in vals.keys():
                vals[x] += y
            else:
                vals[x] = y

    # Sort the dictionary by x
    vals = sorted(vals.items(), key=lambda p: p[0])
    # Put the values back into a dictionary and scale the y-values
    vals = {k: div * v for k, v in vals}
    # Plot the end result
    plt.plot(vals.keys(), vals.values())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def normal_pdf(data: list[int | float]) -> list[list]:
    """
    Returns the x and y values of the normal distribution formed from the
    given data
    :param data: List of numbers
    :return:
    """
    m = mean(data=data)
    dev = std(data)
    data_sort = misc.quicksort(array=data)
    n = 1 / (dev * math.sqrt(2 * math.pi))
    vals = [n * math.exp(-.5 * math.pow((x - m) / dev, 2)) for x in data_sort]
    return [data_sort, vals]


def normal_cdf(xnorm: list[int | float], ynorm: list[int | float],
               a: int | float = None, b: int | float = None) -> int | float:
    """
    Returns the value of the cumulative distribution at x based
    :param xnorm:
    :param ynorm:
    :param a:
    :param b:
    :return:
    """
    # Check that the x values are sorted
    diff = [xnorm[i] - xnorm[i - 1] for i in range(1, len(xnorm))]
    if any(d < 0 for d in diff):
        raise ValueError("X values must be strictly increasing")
    # X and y should be of equal length
    if len(xnorm) != len(ynorm):
        msg = f"X and Y arrays must be of same length. Now got {len(xnorm)} " \
              f"for x, and {len(ynorm)} for y."
        raise ValueError(msg)
    # Set the integration limits
    if a is not None and b is not None and a > b:
        msg = "Lower integration limit can't be smaller than the upper limit."
        raise ValueError(msg)
    if b is None:
        return misc.trapezoid(x=xnorm, y=ynorm, b=a)
    return misc.trapezoid(x=xnorm, y=ynorm, a=a, b=b)


def exp_pdf(x: int | float | list[int | float], lamda: int | float) \
        -> int | float | list[int | float]:
    """
    Probability density function for a exponential distribution of form
    f(x, lamda) = lamda * exp(-lamda * x)
    :param x:
    :param lamda:
    :return:
    """
    if lamda < 0:
        raise ValueError("The parameter lamda must be positive.")
    if type(x) in [int, float]:
        return lamda * math.exp(-lamda * x)
    return [lamda * math.exp(-lamda * xv) for xv in x]


def exp_cdf(x: int | float | list[int | float], lamda: int | float) \
        -> int | float | list[int | float]:
    """
    Cumulative distribution function for a exponential distribution.
    The cdf is of form F(x, lamda) = 1 - exp(-lamda * x)
    :param x:
    :param lamda:
    :return:
    """
    if lamda < 0:
        raise ValueError("The parameter lamda must be positive.")
    if type(x) in [int, float]:
        return 1 - math.exp(-lamda * x)
    return [1 - math.exp(-lamda * xv) for xv in x]


def exp_cdf_inv(y: int | float | list[int | float], lamda: int | float) \
        -> int | float | list[int | float]:
    """
    The inverse function of the cumulative distribution function of an
    exponential distribution
    :param y:
    :param lamda:
    :return:
    """
    if lamda < 0:
        raise ValueError("The parameter lamda must be positive.")
    if type(y) in [int, float]:
        return math.log(1 - y) / -lamda
    return [math.log(1 - yv) / -lamda for yv in y]


def inversion_sample(cdf_inv: ft.partial, n: int) -> list[int | float]:
    """
    Samples n random numbers from the given probability distribution using the
    inversion method. Expects the distribution function (the inverse of the cdf)
    to be given as functools.partial function so that some of the parameters are
    fixed.
    :param cdf_inv:
    :param n:
    :return:
    """
    yinv = [random.random() for _ in range(n)]
    return cdf_inv(y=yinv)


def _randrange(a: int | float, b: int | float) -> float:
    """
    Return a random number from the range [a, b)
    :param a:
    :param b:
    :return:
    """
    return (b - a) * random.random() + a


def rejection_sample(x: list[int | float], pdf: ft.partial, n: int) \
        -> list[int | float]:
    """
    Samples n random numbers from the given probability distribution using the
    rejection method.
    :param x:
    :param pdf:
    :param n:
    :return:
    """
    ylimit = max(pdf(x=x))
    xmin, xmax = min(x), max(x)
    xvals = []
    i = 0
    while i < n:
        xrand = _randrange(a=xmin, b=xmax)
        yrand = _randrange(a=0, b=ylimit)
        ypdf = pdf(x=xrand)
        if yrand <= ypdf:
            xvals.append(xrand)
            i += 1

    return xvals


def main() -> None:
    # Some kind of an example of something
    prices = read_csv(fname="data/hinta.csv", columns=1, delim=";")
    faith = read_csv(fname="data/old_faithful.csv", columns=2)
    kde(data=faith, kernel=kernels.gaussian, xlabel="Waittime [some unit]",
        ylabel="Probability")
    histogram(data=prices, xlabel="Electricity price [c/kWh]",
              ylabel="Probability")
    plt.show()


if __name__ == "__main__":
    main()
