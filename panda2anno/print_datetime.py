import datetime
import json
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import pandas
from pandaset import DataSet

from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="PandaSetの各シーンの日時を出力します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output", type=Path, required=True, help="出力先")
    parser.add_argument("--camera", type=str, default="front_camera", required=False, help="出力対象のcamera")
    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def get_datetime_from_json(timestamps_json: Path) -> str:
    with timestamps_json.open() as f:
        data = json.load(f)
    first_timestamp = data[0]
    first_datetime = datetime.datetime.fromtimestamp(first_timestamp)
    return first_datetime.isoformat()


def main() -> None:
    args = parse_args()
    set_default_logger()

    input_dir: Path = args.input_dir

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    camera_name: str = args.camera
    data: list[dict[str, Any]] = []
    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のsemsegのtimestamp情報を取得します。")
        try:
            str_datetime = get_datetime_from_json(input_dir / sequence_id / "camera" / camera_name / "timestamps.json")

            data.append({"sequence_id": sequence_id, "datetime": str_datetime})

        except Exception:
            logger.warning(f"{sequence_id=}のcuboidのロードに失敗しました。", exc_info=True)
        finally:
            dataset.unload(sequence_id)

    df = pandas.DataFrame(data)
    df.fillna(0, inplace=True)

    tmp_columns = set(df.columns)
    columns = ["sequence_id"] + list(tmp_columns - {"sequence_id"})

    output: Path = args.output
    output.parent.mkdir(exist_ok=True, parents=True)

    df[columns].to_csv(str(output), index=False)


if __name__ == "__main__":
    main()
