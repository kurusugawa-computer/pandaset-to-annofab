import json
import logging
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import numpy
import pandas
from annofab_3dpc.annotation import (
    CuboidAnnotationDetailDataV2,
    CuboidDirection,
    CuboidShapeV2,
    EulerAnglesZXY,
    Location,
    Size,
    Vector3,
)
from pandaset import DataSet
from pandaset.sequence import Sequence
from pyquaternion import Quaternion

from panda2anno.common.pose import Pose
from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


class Cuboid2Annofab:
    def __init__(self, sampling_step: int = 1) -> None:
        self.sampling_step = sampling_step

    def get_direction(self, euler_angle: EulerAnglesZXY) -> CuboidDirection:
        q = Quaternion(euler_angle.to_quaternion())
        rotation_matrix = q.rotation_matrix
        before_front = numpy.array([1, 0, 0]).T
        before_up = numpy.array([0, 0, 1]).T
        front = (rotation_matrix @ before_front).flatten()
        up = (rotation_matrix @ before_up).flatten()
        return CuboidDirection(front=Vector3(front[0], front[1], front[2]), up=Vector3(up[0], up[1], up[2]))

    def get_annotation_detail(self, cuboid: dict[str, Any], lidar_pose: Pose) -> dict[str, Any]:
        """1個のcuboidに対応するAnnofabのアノテーションを取得します。"""

        # lidar座標系のpositionを取得する
        position_in_lidar_coordinate = lidar_pose.inverse() * numpy.array(
            [[cuboid["position.x"], cuboid["position.y"], cuboid["position.z"]]]
        )

        # X軸に対するZ軸の回転角度
        tmp_yaw, _, _ = lidar_pose.inverse().rotation.yaw_pitch_roll
        # pandasetのyawはY軸に対するyawなので、math.pi/2を加える
        yaw = tmp_yaw + cuboid["yaw"] + math.pi / 2

        rotation = EulerAnglesZXY(0, 0, yaw)
        cuboid_data = CuboidAnnotationDetailDataV2(
            CuboidShapeV2(
                dimensions=Size(
                    width=cuboid["dimensions.x"],
                    height=cuboid["dimensions.z"],
                    depth=cuboid["dimensions.y"],
                ),
                location=Location(
                    position_in_lidar_coordinate[0][0],
                    position_in_lidar_coordinate[0][1],
                    position_in_lidar_coordinate[0][2],
                ),
                #
                rotation=rotation,
                direction=self.get_direction(rotation),
            )
        )

        def get_value_or_empty(value: Any) -> str:
            if value is None:
                return ""
            elif isinstance(value, float) and numpy.isnan(value):
                return ""
            return str(value)

        attributes = {
            "object_motion": get_value_or_empty(cuboid["attributes.object_motion"]),
            "rider_status": get_value_or_empty(cuboid["attributes.rider_status"]),
            "pedestrian_behavior": get_value_or_empty(cuboid["attributes.pedestrian_behavior"]),
            "pedestrian_age": get_value_or_empty(cuboid["attributes.pedestrian_age"]),
        }
        result = {
            "annotation_id": cuboid["uuid"],
            "label": cuboid["label"],
            "attributes": attributes,
            "data": cuboid_data.dump()
        }
        return result

    def write_cuboid_annotation_json(self, cuboid_data: pandas.DataFrame, lidar_pose: Pose, output_file: Path):
        cuboid_list = cuboid_data.to_dict("records")
        annotation_details = [self.get_annotation_detail(cuboid, lidar_pose) for cuboid in cuboid_list]

        with output_file.open(mode="w") as f:
            json.dump({"details": annotation_details}, f)

    def write_cuboid_annotations(
        self,
        sequence: Sequence,
        output_dir: Path,
        task_id: str,
    ):
        def get_input_data(index: int) -> str:
            return f"{task_id}-{str(index)}"

        output_dir.mkdir(exist_ok=True, parents=True)
        sequence.load_lidar()

        range_obj = range(0, len(sequence.lidar.data), self.sampling_step)

        sequence.load_cuboids()

        for index in range_obj:
            filename = f"{get_input_data(index)}.json"

            cuboid_data = sequence.cuboids.data[index]

            dict_lidar_pose = sequence.lidar.poses[index]
            self.write_cuboid_annotation_json(
                cuboid_data, lidar_pose=Pose.from_pandaset_pose(dict_lidar_pose), output_file=output_dir / filename
            )


def parse_args():
    parser = ArgumentParser(
        description="PandaSetのcuboidをAnnofabのアノテーションフォーマットに変換します。`annofabcli annotation import`コマンドでインポートすることを想定しています。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="出力先ディレクトリ")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")
    parser.add_argument("--sampling_step", type=int, default=1, required=False, help="指定した値ごとにフレームを出力します。")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_default_logger()

    output_dir: Path = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    input_dir: Path = args.input_dir
    logger.info(f"{input_dir} をKITTIに変換して、{output_dir}にAnnofabのアノテーションを出力します。")

    main_obj = Cuboid2Annofab(sampling_step=args.sampling_step)

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のcuboidをAnnofabのアノテーションに変換します。")
        try:
            main_obj.write_cuboid_annotations(sequence, output_dir=output_dir / sequence_id, task_id=sequence_id)
        except Exception:
            logger.warning(f"{sequence_id=}のcuboidをAnnofabのアノテーションへの変換に失敗しました。", exc_info=True)
        finally:
            dataset.unload(sequence_id)


if __name__ == "__main__":
    main()
