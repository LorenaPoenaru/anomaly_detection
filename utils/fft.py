# from
# https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/fft
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LocalOutlier:
    index: int
    z_score: float

    @property
    def sign(self) -> int:
        return np.sign(self.z_score)


@dataclass
class RegionOutlier:
    start_idx: int
    end_idx: int
    score: float


@contextmanager
def nested_break():
    class NestedBreakException(Exception):
        pass

    try:
        yield NestedBreakException
    except NestedBreakException:
        pass


def reduce_parameters(f: np.ndarray, k: int) -> np.ndarray:
    """
    :param f: fourier transform
    :param k: number of parameters to use
    :return: fourier transform value reduced to k parameters (including the zero frequency term)
    """
    transformed = f.copy()
    if k == 1:
        transformed[1:] = .0
    else:
        transformed[k:-(k - 1)] = 0
    return transformed


def series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - \
        filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


def calculate_local_outlier(
        data: np.ndarray, k: int, c: int, threshold: float) -> List[LocalOutlier]:
    """
    :param data: input data (1-dimensional)
    :param k: number of parameters to be used in IFFT
    :param c: lookbehind and lookahead size for neighbors
    :param threshold: outlier threshold
    :return: list of local outliers
    """
    n = len(data)
    k = max(min(k, n), 1)
    # Fourier transform of data
    y = reduce_parameters(np.fft.fft(data), k)
    f2 = np.real(np.fft.ifft(y))

    # difference of actual data value and the fft fitted curve
    so = np.abs(f2 - data)
    # average difference
    mso = np.mean(so)

    scores = []
    score_idxs = []
    for i in range(n):
        # if the difference at particular point > the average difference
        if so[i] > mso:
            # average value of 'c' neighbors on both sides
            nav = np.average(data[max(i - c, 0):min(i + c, n - 1)])
            # add the local difference (difference of the point and its
            # neighbors) to the collection
            scores.append(data[i] - nav)
            # add the index of suspected outlier to the collection
            score_idxs.append(i)
    scores = np.array(scores)

    #  find average and standard deviation of local difference
    ms = np.mean(scores)
    sds = np.std(scores)

    results = []
    for i in range(len(scores)):
        # calculate the difference between local difference and mean of local difference and divide this by standard
        # deviation of local difference
        z_score = (scores[i] - ms) / sds
        # declare this as an outlier if greater than threshold
        if abs(z_score) > threshold:
            index = score_idxs[i]
            results.append(LocalOutlier(index, z_score))
    return results


def calculate_region_outlier(sign_of_z_result: List[LocalOutlier], max_region: int, max_local_diff: int) -> List[
        RegionOutlier]:
    """
    :param sign_of_z_result: list of local outliers with their z_score
    :param max_region_: maximum outlier region length
    :param max_local_diff: maximum difference between two closed oppositely signed outliers
    :return: list of region outliers
    """

    def next_opposite_sign(index, allowed_diff, data):
        for pos in range(index, min(len(data), index + allowed_diff)):
            if data[index].sign != data[pos].sign:
                return True
        return False

    regions = []
    count = len(sign_of_z_result)
    for i in range(1, count):
        sign = sign_of_z_result[i].sign
        m = 0
        while m < max_local_diff:
            if i >= count:
                break
            if (sign != sign_of_z_result[i].sign):
                # mark i as start of outlier region
                start_idx = i
                i += 1
                n = 0
                while n < max_region:
                    if i < count - 1 and \
                            sign_of_z_result[i].sign == sign_of_z_result[i + 1].sign and \
                            next_opposite_sign(i, max_local_diff, sign_of_z_result):
                        end_idx = i
                        regions.append(RegionOutlier(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            score=np.mean(
                                [abs(l.z_score) for l in sign_of_z_result[start_idx: end_idx + 1]])
                        ))
                        m = m + 1
                        break
                    else:
                        i = i + 1
                        n = n + 1
            else:
                i = i + 1
                m = m + 1

    return regions


def detect_anomalies(data: np.ndarray,
                     ifft_parameters: int = 5,
                     local_neighbor_window: int = 21,
                     local_outlier_threshold: float = .6,
                     max_region_size: int = 50,
                     max_sign_change_distance: int = 10,
                     **args) -> np.ndarray:
    """
    :param data: input time series
    :param ifft_parameters: number of parameters to be used in IFFT
    :param local_neighbor_window: centered window of neighbors to consider for z_score calculation
    :param local_outlier_threshold: outlier threshold in multiples of sigma
    :param max_region_size: maximum outlier region length
    :param max_sign_change_distance: maximum difference between two closed oppositely signed outliers
    :return: anomaly scores (same shape as input)
    """
    neighbor_c = local_neighbor_window // 2
    # print(ifft_parameters, neighbor_c, local_outlier_threshold, max_region_size, max_sign_change_distance)
    local_outliers = calculate_local_outlier(
        data, ifft_parameters, neighbor_c, local_outlier_threshold)
    # print(f"Found {len(local_outliers)} local outliers")

    regions = calculate_region_outlier(
        local_outliers,
        max_region_size,
        max_sign_change_distance)
    # print("Regions: ", regions)

    # broadcast region scores to data points
    anomaly_scores = np.zeros_like(data)
    for reg in regions:
        start_local = local_outliers[reg.start_idx]
        end_local = local_outliers[reg.end_idx]
        anomaly_scores[start_local.index:end_local.index +
                       1] = [reg.score] * (end_local.index - start_local.index + 1)

    import matplotlib.pyplot as plt
    plt.Figure()
    plt.title("Anomaly region scores")
    plt.plot(range(len(data)), data, label="Data")
    plt.plot(range(len(data)), anomaly_scores, label="Anomaly Scores")
    plt.legend()
    plt.savefig('example.png')

    return anomaly_scores
