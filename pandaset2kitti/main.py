import logging
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Optional

import numpy
import pandas
from pandaset import DataSet
from pandaset.sensors import Intrinsics
from pandaset.sequence import Sequence
from pyquaternion import Quaternion

from pandaset2kitti.common.camera import get_camera_matrix_from_intrinsics
from pandaset2kitti.common.kitti import XYZ, CameraViewSettings, KittiImageSeries, KittiVelodyneSeries
from pandaset2kitti.common.kitti import Scene as KittiScene
from pandaset2kitti.common.pose import Pose
from pandaset2kitti.common.utils import set_default_logger

logger = logging.getLogger(__name__)


class Pandaset2Kitti:
    def write_velodyne_bin_file(self, lidar_data: pandas.DataFrame, output_file: Path) -> None:
        """
        LiDARの点群データを、KITTIのvelodyne bin fileに出力する。

        KIITIのvelodyneファイルのフォーマット
        https://github.com/yanii/kitti-pcl/blob/3b4ebfd49912702781b7c5b1cf88a00a8974d944/KITTI_README.TXT#L51-L67
        変換方法
        (N,3) -> (N,4) -> (1,M)

        Args:
            point_cloud: Point Cloud情報
            output_file: 出力ファイル（velodyneのbinファイル）
        """

        df = lidar_data[["x", "y", "z", "i"]]
        # 1次元の配列に変換する
        flatten_data = df.values.flatten().astype(numpy.float32)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        flatten_data.tofile(str(output_file))

    def write_calibration_file(
        self,
        lidar_pose: Pose,
        camera_pose: Pose,
        camera_intrinsics: Intrinsics,
        output_file: Path,
    ) -> None:
        """
        KITTIのcalibration ファイルを生成する。

        Args:
            camera_intrinsics: カメラの内部パラメータ
            camera_pose: World座標系に対するCameraのpose
            point_cloud_pose: World座標系に対するLiDar Sensorのpose
            output_file: 出力先

        """
        P2 = numpy.zeros((3, 4))
        P2[:3, :3] = get_camera_matrix_from_intrinsics(camera_intrinsics)

        R0_rect = numpy.eye(3)

        # 3x4 matrix
        Tr_velo_to_cam = (camera_pose.inverse()).matrix[:3, :]

        output_file.parent.mkdir(exist_ok=True, parents=True)
        with output_file.open(mode="w") as f:
            f.write(f"P2: {' '.join([str(elem) for elem in P2.flatten()])}\n")
            f.write(f"R0_rect: {' '.join([str(elem) for elem in R0_rect.flatten()])}\n")
            f.write(f"Tr_velo_to_cam: {' '.join([str(elem) for elem in Tr_velo_to_cam.flatten()])}\n")

    def write_scene_meta_file(
        self,
        id_list: list[str],
        velodyne_dirname: str,
        kitti_images: list[KittiImageSeries],
        output_file: Path,
    ):

        """
        scene.meta ファイル（annofab-3dpc-editor-cli が読み込むファイル）

        Args:
            id_list:
            image_dir_for_label: labelに対応するimage_dir
            output_scene_dir:

        Returns:

        """
        velodyne = KittiVelodyneSeries(velodyne_dir=velodyne_dirname)
        scene = KittiScene(id_list=id_list, velodyne=velodyne, images=kitti_images, labels=[])
        scene.encode(str(output_file))

    def get_camera_view_setting(
        self,
        camera_pose: Pose,
        camera_intrinsics: Intrinsics,
    ) -> CameraViewSettings:
        """
        3dpc-editorに視野角を表示するために必要な情報を生成。

        Args:
            camera_intrinsics: カメラの内部パラメータ
            camera_pose: World座標系に対するCameraのpose
            point_cloud_pose: World座標系に対するLiDar Sensorのpose

        Returns:
            CameraViewSettings
        """
        fov_x = 2 * math.atan(camera_intrinsics.cx / camera_intrinsics.fx)
        # # sensorからworld座標に変換する行列
        # P_WS = geoPose.from_pose_proto(point_cloud_pose)
        # # cameraからworld座標に変換する行列
        # P_WC = geoPose.from_pose_proto(camera_pose)

        # sensorからcamera座標に変換する行列
        P_CS = camera_pose.inverse()
        # x軸を中心に-90度回転して、LiDar座標系のz軸の向きを合わせる
        tmp_quaterion = Pose(wxyz=Quaternion(axis=[1, 0, 0], angle=-math.pi / 2).q)
        tmp_PC_CS = tmp_quaterion * P_CS
        yaw, _, _ = tmp_PC_CS.rotation.yaw_pitch_roll
        # 3dpc editorはx軸を進行方向としているが、LiDarはy軸が進行方向になっているので、90度回転させる
        # yawの回転軸が逆になっている。。。？
        direction = -yaw + math.pi / 2

        return CameraViewSettings(
            fov=fov_x,
            direction=direction,
            position=XYZ(
                camera_pose.translation[0],
                camera_pose.translation[1],
                camera_pose.translation[2],
            ),
        )

    def write_kitti_scene(
        self,
        sequence: Sequence,
        output_dir: Path,
        sampling_step: int = 1,
        filename_prefix: str = "",
        camera_name_list: Optional[list[str]] = None,
    ):
        def get_filename_stem(index: int) -> str:
            return f"{filename_prefix}{str(index)}"

        sequence.load_lidar()

        range_obj = range(0, len(sequence.lidar.data), sampling_step)

        # 各ファイルの名前（拡張子を除く）に使うID
        id_list = [str(e) for e in range_obj]

        # 点群データの出力
        velodyne_dir = output_dir / "velodyne"
        velodyne_dir.mkdir(exist_ok=True, parents=True)

        for index in range_obj:
            filename = f"{get_filename_stem(index)}.bin"
            lidar_data = sequence.lidar.data[index]
            self.write_velodyne_bin_file(lidar_data, output_file=velodyne_dir / filename)

        # カメラ画像とキャリブレーションファイルの出力
        sequence.load_camera()
        kitti_images = []

        FILE_EXTENSION = "jpg"

        # Annofabで表示する補助画像の順番が自然になるようにする
        if camera_name_list is None:
            camera_name_list = [
                "front_camera",
                "front_left_camera",
                "front_right_camera",
                "left_camera",
                "right_camera",
                "back_camera",
            ]

        for camera_name in camera_name_list:
            if camera_name not in sequence.camera:
                logger.warning(f"{camera_name}の情報は存在しません。")
                continue

            camera_obj = sequence.camera[camera_name]

            calibration_dir = output_dir / f"calib-{camera_name}"
            calibration_dir.mkdir(exist_ok=True, parents=True)
            image_dir = output_dir / f"image-{camera_name}"
            image_dir.mkdir(exist_ok=True, parents=True)

            for index in range_obj:
                pillow_image_obj = camera_obj.data[index]
                dict_lidar_pose = sequence.lidar.poses[0]
                dict_camera_pose = camera_obj.poses[0]

                calibration_filename = f"{get_filename_stem(index)}.txt"
                self.write_calibration_file(
                    lidar_pose=Pose.from_pandaset_pose(dict_lidar_pose),
                    camera_pose=Pose.from_pandaset_pose(dict_camera_pose),
                    camera_intrinsics=camera_obj.intrinsics,
                    output_file=calibration_dir / calibration_filename,
                )

                image_filename = f"{get_filename_stem(index)}.{FILE_EXTENSION}"
                pillow_image_obj.save(str(image_dir / image_filename))

            # 先頭のカメラposeを取得する
            camera_view_setting = self.get_camera_view_setting(
                camera_pose=Pose.from_pandaset_pose(camera_obj.poses[0]), camera_intrinsics=camera_obj.intrinsics
            )
            kitti_images.append(
                KittiImageSeries(
                    image_dir=image_dir.name,
                    calib_dir=calibration_dir.name,
                    file_extension=FILE_EXTENSION,
                    camera_view_setting=camera_view_setting,
                )
            )

        # 拡張KITTI形式用のメタファイルを出力
        self.write_scene_meta_file(
            id_list=id_list,
            velodyne_dirname=velodyne_dir.name,
            kitti_images=kitti_images,
            output_file=output_dir / "scene.meta",
        )


def convert_pantadaset_to_kitti(pandaset_dir: Path, output_dir: Path, sequence_id_list: Optional[list[str]] = None):
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"{pandaset_dir} をKITTIに変換して、{output_dir}に出力します。")

    main_obj = Pandaset2Kitti()

    dataset = DataSet(str(pandaset_dir))

    if sequence_id_list is None:
        sequence_id_list = dataset.sequences()

    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}をKITTIに変換します。")
        main_obj.write_kitti_scene(sequence, output_dir=output_dir / sequence_id, filename_prefix=f"{sequence_id}-")


def parse_args():
    parser = ArgumentParser(
        description="PandaSetを拡張KITTI形式に変換します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="出力先ディレクトリ")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")
    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_default_logger()

    convert_pantadaset_to_kitti(
        args.input_dir,
        output_dir=args.output_dir,
        sequence_id_list=args.sequence_id,
    )


if __name__ == "__main__":
    main()
