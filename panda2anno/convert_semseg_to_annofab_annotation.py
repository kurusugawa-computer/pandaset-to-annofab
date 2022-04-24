import json
import logging
import uuid
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas
from annofab_3dpc.annotation import SegmentAnnotationDetailData, SegmentData
from pandaset import DataSet
from pandaset.sequence import Sequence

from panda2anno.common.annofab import get_input_data_id_from_pandaset
from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


class Semseg2Annofab:
    def __init__(self, sampling_step: int = 1) -> None:
        self.sampling_step = sampling_step

    def write_semseg_annotation_json(
        self, semseg_data: pandas.DataFrame, semseg_classes: dict[int, str], task_dir: Path, input_data_id: str
    ):
        input_data_dir = task_dir / input_data_id
        input_data_dir.mkdir(exist_ok=True, parents=True)

        annotation_details = []
        for class_id in semseg_data["class"].unique():
            df_by_label = semseg_data[semseg_data["class"] == class_id]
            af_segment = SegmentData(list(df_by_label.index))
            annotation_id = str(uuid.uuid4())

            # セグメントファイルを出力
            segment_file = input_data_dir / f"{annotation_id}"
            with segment_file.open("w") as f:
                f.write(af_segment.to_json())

            annotation_details.append(
                {
                    "annotation_id": annotation_id,
                    "label": semseg_classes[str(class_id)],
                    "data": SegmentAnnotationDetailData(data_uri=f"{input_data_id}/{annotation_id}").dump(),
                }
            )

        input_data_json = task_dir / f"{input_data_id}.json"
        with input_data_json.open(mode="w") as f:
            json.dump({"details": annotation_details}, f)

    def write_semseg_annotations(
        self,
        sequence: Sequence,
        output_dir: Path,
        sequence_id: str,
    ):
        output_dir.mkdir(exist_ok=True, parents=True)
        sequence.load_lidar()

        range_obj = range(0, len(sequence.lidar.data), self.sampling_step)

        sequence.load_semseg()

        for index in range_obj:
            input_data_id = get_input_data_id_from_pandaset(sequence_id, index)

            semseg_data = sequence.semseg.data[index]

            self.write_semseg_annotation_json(
                semseg_data, semseg_classes=sequence.semseg.classes, task_dir=output_dir, input_data_id=input_data_id
            )


def parse_args():
    parser = ArgumentParser(
        description="PandaSetのsemantic segmentation をAnnofabのアノテーションフォーマットに変換します。"
        "`annofabcli annotation import`コマンドでインポートすることを想定しています。",
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
    logger.info(f"{input_dir} のSemantic Segmentationを、Annofabのアノテーションフォーマットに変換します。")

    main_obj = Semseg2Annofab(sampling_step=args.sampling_step)

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のsemantic segmentationをAnnofabのアノテーションフォーマットに変換します。")
        try:
            main_obj.write_semseg_annotations(sequence, output_dir=output_dir / sequence_id, sequence_id=sequence_id)
        except Exception:
            logger.warning(f"{sequence_id=}のsemantic segmentationをAnnofabのアノテーションフォーマットに変換に失敗しました。", exc_info=True)
        finally:
            dataset.unload(sequence_id)


if __name__ == "__main__":
    main()
