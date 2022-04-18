import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy

from dgp.proto.annotations_pb2 import AnnotationType, BoundingBox2DAnnotations, BoundingBox3DAnnotations
from dgp.proto.dataset_pb2 import Dataset, Ontology
from dgp.proto.geometry_pb2 import CameraIntrinsics, Pose
from dgp.proto.sample_pb2 import Datum, Sample
from dgp.utils.protobuf import open_pbobject

logger = logging.getLogger(__name__)


class DatasetAccessor:
    _dict_sample: Optional[Dict[str, Sample]] = None

    def __init__(self, dataset_json: Path):
        self.base_dir = dataset_json.parent
        self.dataset = open_pbobject(str(dataset_json), Dataset)
        self.dict_datum = {datum.key: datum for datum in self.dataset.data}

    def get_datum_from_key(self, key: str) -> Optional[Datum]:
        return self.dict_datum.get(key)

    def _create_dict_sample(self):
        return {
            sample.id.name: sample
            for scene_index in self.dataset.scene_splits
            for scene in self.dataset.scene_splits[scene_index].scenes
            for sample in scene.samples
        }

    def get_sample_from_id_name(self, sample_id_name: str) -> Optional[Sample]:
        if self._dict_sample is not None:
            return self._dict_sample.get(sample_id_name)
        else:
            self._dict_sample = self._create_dict_sample()
            return self._dict_sample.get(sample_id_name)

    def get_first_scenes(self):
        """
        `scene_splits` の最初の要素を取得する。
        """
        return self.dataset.scene_splits[next(iter(self.dataset.scene_splits))].scenes

    def get_point_cloud_datum(self, datum_keys: List[str]) -> Datum:
        for key in datum_keys:
            datum = self.get_datum_from_key(key)
            if datum is not None and datum.datum.HasField("point_cloud"):
                return datum
        raise RuntimeError(f"datum_keys={datum_keys} に PointCloud情報は存在しませんでした。")

    def get_calibration(self, calibration_key: str, name: str) -> Tuple[Pose, CameraIntrinsics]:
        """
        キャリブレーション情報の外部パラメータ、内部パラメータを取得します。

        Args:
            calibration_key:
            name: LIDARやカメラの名前

        Returns:
            Tuple(extrinsic, intrinsic)
        """
        calibration = self.dataset.calibration_table[calibration_key]
        try:
            index = list(calibration.names).index(name)
            return (calibration.extrinsics[index], calibration.intrinsics[index])
        except ValueError as e:
            raise RuntimeError(
                f"calibration_key='{calibration_key}'のキャリブレーション情報のnames={calibration.names} に {name}が含まれていません。"
            ) from e

    def get_image_data(self, datum_keys: List[str]) -> List[Datum]:
        data = []
        for key in datum_keys:
            datum = self.get_datum_from_key(key)
            if datum is not None and datum.datum.HasField("image"):
                data.append(datum)

        return data

    def get_image_datum_with_id_name(self, datum_keys: List[str], id_name: str) -> Optional[Datum]:
        for key in datum_keys:
            datum = self.get_datum_from_key(key)
            if datum is not None and datum.datum.HasField("image"):
                if datum.id.name == id_name:
                    return datum

        return None

    def get_path(self, filename: str) -> Path:
        """
        dataset_v1.jsonに記載されているパスから、フルパスを取得する。
        """
        return self.base_dir / filename

    def read_point_cloud_file(self, filename: str) -> numpy.ndarray:
        """
        pointcloudが格納されたnpzファイルを読み込む。
        npzファイル内には１つのnpyファイルが格納されていること前提。

        Args:
            filename: npzファイルのパス。dataset.jsonの存在するディレクトリからのパスを指定する。

        Returns:
            numpy array
        """
        with numpy.load(str(self.base_dir / filename)) as f:
            data = f[f.files[0]]
            return data

    def read_bounding_box_3d_json(self, annotations):
        if AnnotationType.BOUNDING_BOX_3D not in annotations:
            return []
        annotation_json = annotations[AnnotationType.BOUNDING_BOX_3D]
        pbobject = open_pbobject(str(self.get_path(annotation_json)), BoundingBox3DAnnotations)
        return pbobject.annotations

    def read_bounding_box_2d_json(self, annotations):
        if AnnotationType.BOUNDING_BOX_2D not in annotations:
            return []
        annotation_json = annotations[AnnotationType.BOUNDING_BOX_2D]
        pbobject = open_pbobject(str(self.get_path(annotation_json)), BoundingBox2DAnnotations)
        return pbobject.annotations

    def read_ontology_file(self, ontology_filename: str):
        return open_pbobject(str(self.get_path(f"ontology/{ontology_filename}.json")), Ontology)
