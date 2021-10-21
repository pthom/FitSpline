import numpy as np
import math
import cv2
from typing import *

Point2D = np.ndarray
PointsArray = np.ndarray

"""
Spline interpolations functions:

Catmull rom is adapted straight from the wikipedia article : 
https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
"""

def _dist_point(p1: Point2D, p2: Point2D) -> float:
    return np.linalg.norm(p2 - p1)


def _get_t(t: float, pt0: Point2D, pt1: Point2D, alpha: float) -> float:
    a = math.pow((pt1[0] - pt0[0]), 2.0) + math.pow((pt1[1] - pt0[1]), 2.0)
    b = math.pow(a, 0.5)
    c = math.pow(b, alpha)
    return c + t


def _interpolate_centripetal_catmull_rom_p1p2(
        p0: Point2D,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        nb_points: int,
        alpha: float) -> PointsArray:
    # if p1 and p2 are the same, we cannot interpolate
    epsilon = 1E-8

    if _dist_point(p1, p2) < epsilon:
        new_points = np.zeros((1, 2))
        new_points[0] = p1
        return new_points
    # if p0 and p1 are too close, we separate them to avoid a div by zero
    if _dist_point(p0, p1) < epsilon:
        p1[0] += 1E-6
    # if p2 and p3 are too close, we separate them to avoid a div by zero
    if _dist_point(p2, p3) < epsilon:
        p3[0] += 1E-6


    new_points = np.zeros((nb_points, 2))

    t0 = 0.0
    t1 = _get_t(t0, p0, p1, alpha)
    t2 = _get_t(t1, p1, p2, alpha)
    t3 = _get_t(t2, p2, p3, alpha)

    dt = (t2 - t1) / nb_points

    i = 0
    t = t1
    while i < nb_points:
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2

        t = t + dt

        new_points[i] = C
        i = i + 1

    return new_points


def interpolate_centripetal_catmull_rom(
        control_points: PointsArray,
        nb_minimum_points: int,
        alpha: float
    ) -> PointsArray:

    nb_points = control_points.shape[0]
    k = float(nb_minimum_points) / (float(nb_points) - 3.)
    nb_points_per_segment = int(k + 1.)

    interpolated_segments = []
    nb_points_total = 0
    for i in range(len(control_points) - 3):
        p0 = control_points[i]
        p1 = control_points[i + 1]
        p2 = control_points[i + 2]
        p3 = control_points[i + 3]

        interpolated_segment = _interpolate_centripetal_catmull_rom_p1p2(p0, p1, p2, p3, nb_points_per_segment, alpha)
        interpolated_segments.append(interpolated_segment)
        nb_points_total = nb_points_total + interpolated_segment.shape[0]

    interpolated_curve = np.zeros((nb_points_total, 2))
    idx = 0
    for interpolated_segment in interpolated_segments:
        for jj in range(interpolated_segment.shape[0]):
            interpolated_curve[idx] = interpolated_segment[jj]
            idx = idx + 1
    return interpolated_curve


def interpolate_centripetal_catmull_rom_circular_shape(
        control_points: PointsArray,
        nb_minimum_points: int,
        alpha: float
    ) -> PointsArray:

    nb_control_points = control_points.shape[0]

    control_points_circular = np.zeros((nb_control_points + 3, 2))
    for i in range(nb_control_points):
        control_points_circular[i] = control_points[i]

    if nb_control_points > 2:
        control_points_circular[nb_control_points     ] = control_points[0]
        control_points_circular[nb_control_points + 1 ] = control_points[1]
        control_points_circular[nb_control_points + 2 ] = control_points[2]
    else:
        control_points_circular[nb_control_points     ] = control_points[0]
        control_points_circular[nb_control_points + 1 ] = control_points[1]
        control_points_circular[nb_control_points + 2 ] = control_points[1]
    return interpolate_centripetal_catmull_rom(control_points_circular, nb_minimum_points, alpha)


def interpolate_pascal_spline_circular_shape(
        control_points: PointsArray,
        nb_minimum_points: int,
        alpha: float
    ) -> PointsArray:

    nb_control_points = control_points.shape[0]
    if nb_control_points == 0:
        e = np.zeros((0, 2))
        return e

    # On precalcule le perimetre:
    # quand des points sont rapproches, on diminuera la valeur alpha
    # (on suppose que l'utilisateur veut faire un angle aigu)
    perimeter = 0.
    for i in range(nb_control_points):
        i1 = (i + 1) if ((i + 1) < nb_control_points) else 0
        perimeter += _dist_point(control_points[i], control_points[i1])

    current_interpolated = control_points
    nb_loops = int(math.log(float(nb_minimum_points) / float(nb_control_points)) / math.log(2.) + 0.5)
    nb_points_current = current_interpolated.shape[0]

    # On double le nombre de points a chaque boucle
    for loopIndex in range(nb_loops):
        # Distance moyenne attendue entre deux points
        normal_dist = perimeter / nb_points_current
        old_interpolated = current_interpolated

        # KK interpolated.resize(nb * 2)
        current_interpolated = np.zeros((nb_points_current * 2, 2))
        for i in range(nb_points_current):
            current_interpolated[i * 2] = old_interpolated[i]

        for i0 in range(nb_points_current):
            i_1 = i0 - 1 if i0 > 0 else nb_points_current - 1
            i1 = i0 + 1 if i0 < nb_points_current - 1 else 0
            i2 = i1 + 1 if i1 < nb_points_current - 1 else 0
            p_1 = old_interpolated[i_1]
            p0 = old_interpolated[i0]
            p1 = old_interpolated[i1]
            p2 = old_interpolated[i2]

            dist = _dist_point(p0, p1)
            dist_1 = _dist_point(p0, p_1)
            dist2 =  _dist_point(p1, p2)

            if dist > .0001:
                if dist_1 > .0001:
                    # Plus la distance est petite, plus alpha est petit
                    # (on suppose que l'utilisateur veut faire un angle aigu)
                    # Distance "normale" = perimeter / nb
                    if dist_1 < normal_dist:
                        current_alpha = alpha * (dist_1 / normal_dist) * (dist_1 / normal_dist)
                    else:
                        current_alpha = alpha

                    u = (p1 - p0) / dist + (current_alpha * (p0 - p_1) / dist_1)
                    k = dist / (np.linalg.norm(u)) / 2.0
                    dst1 = p0 + (k * u)
                else:
                    dst1 = (p0 + p1) / 2.

                if dist2 > .0001:
                    # Plus la distance est petite, plus alpha est petit
                    # (on suppose que l'utilisateur veut faire un angle aigu)
                    # Distance "normale" = perimeter / nb
                    normal_dist = perimeter / nb_points_current
                    if dist2 < normal_dist:
                        current_alpha = alpha * (dist2 / normal_dist) * (dist2 / normal_dist)
                    else:
                        current_alpha = alpha

                    v = (p0 - p1) / dist + (current_alpha * (p1 - p2) / dist2)
                    k = dist / (np.linalg.norm(v)) / 2.0
                    dst2 = p1 + (k * v)
                else:
                    dst2 = (p0 + p1) / 2.

                current_interpolated[2 * i0 + 1] = (dst1 + dst2) / 2.

            else:
                current_interpolated[2 * i0 + 1] = p0
        nb_points_current = nb_points_current * 2
    return current_interpolated


import time


def typical_shapes() -> Dict[str, PointsArray]:
    r = {}
    r["Pantos"] = np.array(
        [
            [622., 349.],
            [488., 464.],
            [265., 466.],
            [152., 346.],
            [131., 198.],
            [177., 87.],
            [402., 50.],
            [659., 133.]
        ]
    )
    r["Square"] = np.array(
        [
            [563., 387.],
            [371, 423],
            [150, 396],
            [94, 275],
            [118, 144],
            [371, 102],
            [588, 151],
            [611, 260]
        ]
    )
    r["Octo"] = np.array(
        [
            [311., 307.],
            [267., 350.],
            [216., 358.],
            [163., 353.],
            [108., 317.],
            [94., 227.],
            [114., 196.],
            [164., 177.],
            [291., 189.],
            [320., 223.]
        ]

    )
    r["Papillon"] = np.array(
        [
            [307., 325.],
            [217., 354.],
            [131., 322.],
            [104., 263.],
            [128., 231.],
            [217., 207.],
            [311., 231.],
            [338., 271.]
        ]
    )

    r["Bifocal_RoundTop"] = np.array(
        [
            [511., 263.],
            [474, 327],
            [406, 349],
            [341, 322],
            [306, 262],
            [300, 226],
            [304, 204],
            [325, 195],
            [360, 190],
            [403, 188],
            [468, 192],
            [512, 206],
            [514, 232]
        ]
    )

    r_interp = {}
    for shape_name, shape_control_points in r.items():
        r_interp[shape_name] = interpolate_pascal_spline_circular_shape(shape_control_points, 300, 0.4)
    return r_interp

