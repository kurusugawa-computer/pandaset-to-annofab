from pandaset.sensors import Intrinsics
import numpy


def get_camera_matrix_from_intrinsics(intrinsics: Intrinsics) -> numpy.ndarray:
    """
    3x3のカメラ内部行列を取得します。
    """
    K = numpy.eye(3)
    K[0, 0] = intrinsics.fx
    K[1, 1] = intrinsics.fy
    K[0, 2] = intrinsics.cx
    K[1, 2] = intrinsics.cy
    return K
