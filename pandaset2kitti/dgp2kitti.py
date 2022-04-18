import logging
import math
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import ClassVar, List, Tuple

import numpy
from PIL import Image
from pyquaternion import Quaternion

from dgp.proto.geometry_pb2 import CameraIntrinsics, Pose
from dgp.proto.point_cloud_pb2 import PointCloud
from dgp.proto.sample_pb2 import Datum, Sample
from dgp.proto.scene_pb2 import Scene
from dgp.utils.camera import camera_matrix_from_pbobject
from dgp.utils.geometry import Pose as geoPose
from dgputils.common.dataset_accessor import DatasetAccessor
from dgputils.common.utils import set_default_logger
from dgputils.kitti.scene import XYZ, CameraViewSettings, KittiImageSeries, KittiVelodyneSeries
from dgputils.kitti.scene import Scene as KittiScene

logger = logging.getLogger(__name__)


class ConvertDgp2Kitti:
    def __init__(self, dataset_accessor: DatasetAccessor):
        self.dataset_accessor = dataset_accessor

    CALIBRATION_DIR_PREFIX: ClassVar[str] = "calib-"
    LABEL_DIR_PREFIX: ClassVar[str] = "label-"

    @staticmethod
    def create_camera_view_settings(
        point_cloud_pose: Pose,
        camera_pose: Pose,
        camera_intrinsics: CameraIntrinsics,
        camera_extrinsics: Pose,
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
        # sensorからworld座標に変換する行列
        P_WS = geoPose.from_pose_proto(point_cloud_pose)
        # cameraからworld座標に変換する行列
        P_WC = geoPose.from_pose_proto(camera_pose)

        # sensorからcamera座標に変換する行列
        P_CS = P_WC.inverse() * P_WS
        # x軸を中心に-90度回転して、LiDar座標系のz軸の向きを合わせる
        tmp_quaterion = geoPose(wxyz=Quaternion(axis=[1, 0, 0], angle=-math.pi / 2).q)
        tmp_PC_CS = tmp_quaterion * P_CS
        yaw, _, _ = tmp_PC_CS.rotation.yaw_pitch_roll
        # 3dpc editorはx軸を進行方向としているが、LiDarはy軸が進行方向になっているので、90度回転させる
        # yawの回転軸が逆になっている。。。？
        direction = -yaw + math.pi / 2

        geo_camera_extrinsics = geoPose.from_pose_proto(camera_extrinsics)
        return CameraViewSettings(
            fov=fov_x,
            direction=direction,
            position=XYZ(
                geo_camera_extrinsics.translation[0],
                geo_camera_extrinsics.translation[1],
                geo_camera_extrinsics.translation[2],
            ),
        )

    def write_velodyne_bin_file(self, point_cloud: PointCloud, output_file: Path) -> None:
        """
        npzから読み込んだ情報を KITTI velodyne bin fileに出力する。

        変換方法
        (N,3) -> (N,4) -> (1,M)

        Args:
            point_cloud: Point Cloud情報
            output_file: 出力ファイル（velodyneのbinファイル）

        """
        point_cloud_data = self.dataset_accessor.read_point_cloud_file(point_cloud.filename)

        point_format = list(point_cloud.point_format)

        try:
            x_column_index = point_format.index(point_cloud.ChannelType.X)
            y_column_index = point_format.index(point_cloud.ChannelType.Y)
            z_column_index = point_format.index(point_cloud.ChannelType.Z)
        except ValueError as e:
            raise RuntimeError(
                f"point_cloud.point_formatにXYZが存在しませんでした。" f"point_format={point_cloud.point_format}"
            ) from e

        point_size = point_cloud_data.shape[0]
        zero_column = numpy.zeros((point_size, 1))
        data = numpy.hstack(
            (
                point_cloud_data[:, [x_column_index, y_column_index, z_column_index]],
                zero_column,
            )
        )
        flatten_data = data.flatten().astype(numpy.float32)
        # 1次元の配列に変換する
        output_file.parent.mkdir(exist_ok=True, parents=True)
        flatten_data.tofile(str(output_file))

    def write_image_file(
        self,
        image_filename: str,
        output_dir: Path,
        frame_name: str,
        convert_to_png: bool = False,
    ) -> None:
        """
        画像ファイルを生成する。

        Args:
            image_filename:
            output_dir:
            convert_to_png: PNGに変換するかどうか。3dpc-editor-cliの都合でPNGに変換した方が都合がよい。

        """
        image_path = self.dataset_accessor.get_path(image_filename)
        output_dir.mkdir(exist_ok=True, parents=True)
        if convert_to_png:
            im = Image.open(image_path)
            output_file = output_dir / f"{frame_name}.png"
            im.save(output_file)
        else:
            output_file = output_dir / f"{frame_name}{Path(image_filename).suffix}"
            shutil.copyfile(image_path, output_file)

    # @classmethod
    # def annotation_to_kitti_label(
    #     cls, annotation: BoundingBox3DAnnotation
    # ) -> KittiLabel:
    #     """
    #     たぶんまちがっている
    #     Args:
    #         annotation:
    #
    #     Returns:
    #
    #     """
    #     pose = geoPose.from_pose_proto(annotation.box.pose)
    #     rm = pose.rotation_matrix
    #     # カメラ座標系のy軸に対して回転する角度（X軸と同じ方向に向いていたら0）
    #     yaw = math.atan2(rm[2][0], rm[0][0])
    #     return KittiLabel(
    #         type=annotation.class_id,
    #         height=annotation.box.height,
    #         width=annotation.box.width,
    #         depth=annotation.box.length,
    #         x=annotation.box.pose.translation.x,
    #         y=annotation.box.pose.translation.y,
    #         z=annotation.box.pose.translation.z,
    #         yaw=yaw,
    #         annotation_id=str(annotation.instance_id),
    #     )

    # @classmethod
    # def write_label_csv(
    #     cls,
    #     annotations,
    #     output_file: Path,
    # ) -> None:
    #     """
    #     アノテーション情報をKITTI label形式で出力する。
    #     labelの情報はカメラ座標系。
    #     ただし角度情報は1つしかないので、正確に角度変換できない。
    #
    #     Args:
    #         point_cloud_pose: World座標系に対するLiDar Sensorのpose
    #         output_file: 出力先
    #
    #     """
    #     kitti_label_list = [cls.annotation_to_kitti_label(anno) for anno in annotations]
    #     KittiLabel.encode_path(kitti_label_list, output_file)

    @staticmethod
    def write_calibration_file(
        point_cloud_pose: Pose,
        camera_pose: Pose,
        camera_intrinsics: CameraIntrinsics,
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
        P2[:3, :3] = camera_matrix_from_pbobject(camera_intrinsics)

        R0_rect = numpy.eye(3)

        P_WS = geoPose.from_pose_proto(point_cloud_pose)
        P_WC = geoPose.from_pose_proto(camera_pose)
        # 3x4 matrix
        Tr_velo_to_cam = (P_WC.inverse() * P_WS).matrix[:3, :]

        output_file.parent.mkdir(exist_ok=True, parents=True)
        with output_file.open(mode="w") as f:
            f.write(f"P2: {' '.join([str(elem) for elem in P2.flatten()])}\n")
            f.write(f"R0_rect: {' '.join([str(elem) for elem in R0_rect.flatten()])}\n")
            f.write(f"Tr_velo_to_cam: {' '.join([str(elem) for elem in Tr_velo_to_cam.flatten()])}\n")

    def write_scene_meta_file(
        self,
        id_list: List[str],
        velodyne_dirname: str,
        image_dirname_list: List[str],
        camera_view_settings_list: List[CameraViewSettings],
        output_scene_dir: Path,
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

        kitti_images = []
        assert len(image_dirname_list) == len(
            camera_view_settings_list
        ), "len(image_dirname_list) と len(camera_view_settings_list) が同じ値でない"
        for image_dirname, camera_view_settings in zip(image_dirname_list, camera_view_settings_list):
            kitti_images.append(
                KittiImageSeries(
                    image_dir=image_dirname,
                    calib_dir=f"{self.CALIBRATION_DIR_PREFIX}{image_dirname}",
                    camera_view_setting=camera_view_settings,
                )
            )

        scene = KittiScene(id_list=id_list, velodyne=velodyne, images=kitti_images, labels=[])
        scene.encode(str(output_scene_dir / "scene.meta"))

    def execute_sample(
        self, sample: Sample, output_scene_dir: Path, convert_to_png: bool = False
    ) -> Tuple[str, Datum, List[Datum], List[CameraViewSettings]]:
        """
        1個のフレームに対して処理を実行する。

        Args:
            sample: シーン情報
            output_scene_dir: シーンの出力ディレクトリ
            convert_to_png:

        Returns:
            Tuple[frame_name, point_cloud_datum, image_data, camera_view_settings_list]
        """

        def get_frame_name(sample, point_cloud_datum) -> str:
            if sample.id.name != "":
                frame_name = sample.id.name
            else:
                # LiDarのファイル名を基準にする
                frame_name = Path(point_cloud_datum.datum.point_cloud.filename).stem
            return frame_name

        datum_keys = sample.datum_keys
        point_cloud_datum = self.dataset_accessor.get_point_cloud_datum(datum_keys)

        image_data = self.dataset_accessor.get_image_data(datum_keys)
        assert len(image_data) > 0, f"{sample.name} image情報が存在しませんでした。datum_keys={datum_keys}"

        frame_name = get_frame_name(sample, point_cloud_datum)
        point_cloud_id_name = point_cloud_datum.id.name
        self.write_velodyne_bin_file(
            point_cloud=point_cloud_datum.datum.point_cloud,
            output_file=output_scene_dir / f"{point_cloud_id_name}/{frame_name}.bin",
        )

        camera_view_settings_list = []
        for image_datum in image_data:
            image_id_name = image_datum.id.name
            self.write_image_file(
                image_filename=image_datum.datum.image.filename,
                output_dir=output_scene_dir / image_id_name,
                frame_name=frame_name,
                convert_to_png=convert_to_png,
            )

            (
                camera_extrinsics,
                camera_intrinsics,
            ) = self.dataset_accessor.get_calibration(sample.calibration_key, image_id_name)

            self.write_calibration_file(
                point_cloud_pose=point_cloud_datum.datum.point_cloud.pose,
                camera_pose=image_datum.datum.image.pose,
                camera_intrinsics=camera_intrinsics,
                output_file=output_scene_dir / f"{self.CALIBRATION_DIR_PREFIX}{image_id_name}/{frame_name}.txt",
            )
            camera_view_settings = self.create_camera_view_settings(
                point_cloud_pose=point_cloud_datum.datum.point_cloud.pose,
                camera_pose=image_datum.datum.image.pose,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
            )
            camera_view_settings_list.append(camera_view_settings)

        return (
            frame_name,
            point_cloud_datum,
            image_data,
            camera_view_settings_list,
        )

    def execute_scene(self, scene: Scene, output_scene_dir: Path, convert_to_png: bool = False) -> None:
        """
        1個のシーンに対して処理を実行する。

        Args:
            scene: シーン情報
            output_scene_dir: シーンの出力ディレクトリ
            convert_to_png:
        """
        if len(scene.samples) == 0:
            logger.warning(f"{scene.name} の `samples`は空です。")
            return

        id_list = []

        point_cloud_datum = None
        image_data = None
        camera_view_settings_list = None
        for sample in scene.samples:
            (
                frame_name,
                point_cloud_datum,
                image_data,
                camera_view_settings_list,
            ) = self.execute_sample(sample, output_scene_dir=output_scene_dir, convert_to_png=convert_to_png)
            id_list.append(frame_name)

        assert point_cloud_datum is not None and image_data is not None and camera_view_settings_list is not None

        velodyne_dirname = point_cloud_datum.id.name
        image_dirname_list = [image_datum.id.name for image_datum in image_data]

        self.write_scene_meta_file(
            id_list,
            velodyne_dirname=velodyne_dirname,
            image_dirname_list=image_dirname_list,
            camera_view_settings_list=camera_view_settings_list,
            output_scene_dir=output_scene_dir,
        )


def convert_dgp_to_kitti(dataset_json: Path, output_dir: Path, convert_to_png: bool = False):
    """
    DGPの`dataset.json`を、KITTI formatに変換する。

    Args:
        dataset_json: DGPの`dataset.json`
        output_dir: 出力先
    """

    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_accessor = DatasetAccessor(dataset_json)
    logger.debug(f"{dataset_json} の読み込み完了")

    main_obj = ConvertDgp2Kitti(dataset_accessor)
    for scene_index in dataset_accessor.dataset.scene_splits:
        scenes = dataset_accessor.dataset.scene_splits[scene_index].scenes
        for scene in scenes:
            logger.debug(f"scene.name={scene.name}をKITTI形式に変換します。")

            try:
                main_obj.execute_scene(
                    scene,
                    output_scene_dir=output_dir / scene.name,
                    convert_to_png=convert_to_png,
                )
            except Exception:  # pylint: disable=broad-except
                logger.warning(f"scene.name={scene.name} のKITTI形式への変換に失敗しました。", exc_info=True)
                continue


def parse_args():
    parser = ArgumentParser(
        description="pointcloudのDGPフォーマットをKITTI形式に変換する。KITTI形式に変換したものをanno3dコマンドに登録する。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_json", type=Path, required=True, help="DGPフォーマットのdataset.json")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="出力先ディレクトリ")
    parser.add_argument(
        "--convert_to_png",
        help="PNG画像に変換する。現在、`anno3d`コマンドはPNG画像にしか対応していないため。",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_default_logger()

    convert_dgp_to_kitti(
        args.dataset_json,
        output_dir=args.output_dir,
        convert_to_png=args.convert_to_png,
    )


if __name__ == "__main__":
    main()
