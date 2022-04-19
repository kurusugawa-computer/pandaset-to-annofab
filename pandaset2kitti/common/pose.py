import numpy
from pyquaternion import Quaternion


class Pose:
    """
    SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.

    https://github.com/TRI-ML/dgp/blob/6ff13df792ac210a7b4e3c2f11c57079ac9c884a/dgp/utils/pose.py を流用しました。
    """

    def __init__(self, wxyz=numpy.float64([1, 0, 0, 0]), tvec=numpy.float64([0, 0, 0])):
        """Initialize a Pose with Quaternion and 3D Position

        Parameters
        ----------
        wxyz: numpy.float64 or Quaternion, (default: numpy.float64([1,0,0,0]))
            Quaternion/Rotation (wxyz)

        tvec: numpy.float64, (default: numpy.float64([0,0,0]))
            Translation (xyz)
        """
        self.quat = Quaternion(wxyz)
        self.tvec = tvec

    def __repr__(self):
        formatter = {"float_kind": lambda x: "%.2f" % x}
        tvec_str = numpy.array2string(self.tvec, formatter=formatter)
        return "wxyz: {}, tvec: ({})".format(self.quat, tvec_str)

    def __mul__(self, other):
        """Left-multiply Pose with another Pose or 3D-Points.

        Parameters
        ----------
        other: Pose or numpy.ndarray
            1. Pose: Identical to oplus operation.
               (i.e. self_pose * other_pose)
            2. ndarray: transform [N x 3] point set
               (i.e. X' = self_pose * X)

        Returns
        ----------
        result: Pose or numpy.ndarray
            Transformed pose or point cloud
        """
        if isinstance(other, Pose):
            assert isinstance(other, self.__class__)
            t = self.quat.rotate(other.tvec) + self.tvec
            q = self.quat * other.quat
            return self.__class__(q, t)
        else:
            assert other.shape[-1] == 3, "Point cloud is not 3-dimensional"
            X = numpy.hstack([other, numpy.ones((len(other), 1))]).T
            return (numpy.dot(self.matrix, X).T)[:, :3]

    def __rmul__(self, other):
        raise NotImplementedError("Right multiply not implemented yet!")

    def inverse(self):
        """Returns a new Pose that corresponds to the
        inverse of this one.

        Returns
        ----------
        result: Pose
            Inverted pose
        """
        qinv = self.quat.inverse
        return self.__class__(qinv, qinv.rotate(-self.tvec))

    @property
    def matrix(self):
        """Returns a 4x4 homogeneous matrix of the form [R t; 0 1]

        Returns
        ----------
        result: numpy.ndarray
            4x4 homogeneous matrix
        """
        result = self.quat.transformation_matrix
        result[:3, 3] = self.tvec
        return result

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: numpy.ndarray
            3x3 rotation matrix
        """
        result = self.quat.transformation_matrix
        return result[:3, :3]

    @property
    def rotation(self):
        """Return the rotation component of the pose as a Quaternion object.

        Returns
        ----------
        self.quat: Quaternion
            Rotation component of the Pose object.
        """
        return self.quat

    @property
    def translation(self):
        """Return the translation component of the pose as a numpy.ndarray.

        Returns
        ----------
        self.tvec: numpy.ndarray
            Translation component of the Pose object.
        """
        return self.tvec

    @classmethod
    def from_matrix(cls, transformation_matrix):
        """Initialize pose from 4x4 transformation matrix

        Parameters
        ----------
        transformation_matrix: numpy.ndarray
            4x4 containing rotation/translation

        Returns
        -------
        Pose
        """
        return cls(
            wxyz=Quaternion(matrix=transformation_matrix[:3, :3]), tvec=numpy.float64(transformation_matrix[:3, 3])
        )

    def __eq__(self, other):
        return self.quat == other.quat and (self.tvec == other.tvec).all()

    @classmethod
    def from_pandaset_pose(cls, pandaset_pose: dict[str, dict[str, float]]) -> "Pose":
        """pandasetのposeから、生成します。

        Args:
            dict_pose: pandasetのpose
        """

        heading = pandaset_pose["heading"]
        rotation = numpy.float64([heading["w"], heading["x"], heading["y"], heading["z"]])  # type: ignore

        position = pandaset_pose["position"]
        translation = numpy.float64([position["x"], position["y"], position["z"]])  # type: ignore
        return cls(wxyz=rotation, tvec=translation)

    @property
    def pandaset_pose(self) -> dict[str, dict[str, float]]:
        """pandaset用のposeに変換します。

        Returns:
            pandaset用のpose
        """
        position = {
            "x": self.tvec[0],
            "y": self.tvec[1],
            "z": self.tvec[2],
        }

        heading = {
            "w": self.quat[0],
            "x": self.quat[1],
            "y": self.quat[2],
            "z": self.quat[3],
        }
        return {
            "position": position,
            "heading": heading,
        }
