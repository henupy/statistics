"""
Some functions that are used in the statistics-related functions, but
are not necessarily related to statistics per say are stashed in here
"""


def _partition(array: list[int | float], lo: int, hi: int) -> int:
    """
    Returns the 'pivot' index
    :param array:
    :param lo:
    :param hi:
    :return:
    """
    mid = (hi - lo) // 2 + lo
    pivot = array[mid]  # The pivot value
    i = lo - 1  # Left index
    j = hi + 1  # Right index
    while True:
        # Move the left index to the right at least once and while the
        # element at the left index is less than the pivot
        while True:
            i += 1
            if array[i] >= pivot:
                break
        # Move the right index to the left at least once and while the
        # element at the left index is less than the pivot
        while True:
            j -= 1
            if array[j] <= pivot:
                break
        # If the indices crossed, return
        if i >= j:
            return j
        # Swap the elements at the left and right indices
        array[i], array[j] = array[j], array[i]


def _quicksort(array: list[int | float], lo: int, hi: int) -> None:
    """
    Quicksort in-place
    :param array:
    :param lo:
    :param hi:
    :return:
    """
    if not (0 <= lo < hi and hi >= 0):
        return
    p = _partition(array=array, lo=lo, hi=hi)
    _quicksort(array=array, lo=lo, hi=p)
    _quicksort(array=array, lo=p + 1, hi=hi)


def quicksort(array: list[int | float], ascending: bool = True) \
        -> list[int | float]:
    """
    A helper function to create a copy of the array, and to take away
    the need to define all the three arguments of the _quicksort function. Also
    adds the possibility to decide whether to sort in ascending or descending
    order.

    The algorihm is purely implemented by following its Wikipedia article:
    https://en.wikipedia.org/wiki/Quicksort
    :param array:
    :param ascending:
    :return:
    """
    arr = array[:]
    _quicksort(array=arr, lo=0, hi=len(arr) - 1)
    if not ascending:
        return arr[::-1]
    return arr


def binary_search(nums: list[int | float], value: int | float) -> int:
    """
    Returns the index that the given value would be needed to insert
    so that the given sorted array of numbers would still be sorted
    :param nums:
    :param value:
    :return:
    """
    low, high = 0, len(nums)
    while low < high:
        mid = (low + high) // 2
        if nums[mid] < value:
            low = mid + 1
        else:
            high = mid

    return low


def clamp(val: int | float, low: int | float, high: int | float) -> int | float:
    """
    Clamps the given value between the lower and higher limits
    :param val:
    :param low:
    :param high:
    :return:
    """
    if val < low:
        return low
    if val > high:
        return high
    return val
