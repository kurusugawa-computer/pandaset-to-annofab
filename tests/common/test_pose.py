from pandaset2kitti.common.pose import Pose

from pytest import approx

pandaset_pose = {
    "position": {"x": 2.5866664556528294, "y": 2.466240979290467, "z": 1.8213244045644146},
    "heading": {"w": 0.6554122851680118, "x": -0.650630103896342, "y": 0.27605590080108616, "z": -0.26628620690449},
}


def test_converting_pandaset_pose():
    pose = Pose.from_pandaset_pose(pandaset_pose)
    actual = pose.pandaset_pose
    assert pandaset_pose == actual


def test_converting_matrix():
    pose = Pose.from_pandaset_pose(pandaset_pose)

    matrix = pose.matrix
    actual = Pose.from_matrix(matrix)
    assert pose == actual


def test_inverse():
    pose = Pose.from_pandaset_pose(pandaset_pose)

    inverse = pose.inverse()
    unit_pose = pose * inverse

    assert list(unit_pose.quat) == approx([1, 0, 0, 0])
    assert unit_pose.tvec == approx([0, 0, 0])
