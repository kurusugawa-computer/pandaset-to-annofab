import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas
from pandaset import DataSet
from pandaset.sequence import Sequence

from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


class AttributeCounter:
    object_motion: Counter
    rider_status: Counter
    pedestrian_behavior: Counter
    pedestrian_age: Counter

    def __init__(
        self,
        object_motion: Optional[Counter] = None,
        rider_status: Optional[Counter] = None,
        pedestrian_behavior: Optional[Counter] = None,
        pedestrian_age: Optional[Counter] = None,
    ):
        self.object_motion = object_motion if object_motion is not None else Counter()
        self.rider_status = rider_status if rider_status is not None else Counter()
        self.pedestrian_behavior = pedestrian_behavior if pedestrian_behavior is not None else Counter()
        self.pedestrian_age = pedestrian_age if pedestrian_age is not None else Counter()

    def __add__(self, obj: "AttributeCounter") -> "AttributeCounter":
        return AttributeCounter(
            object_motion=self.object_motion + obj.object_motion,
            rider_status=self.rider_status + obj.rider_status,
            pedestrian_behavior=self.pedestrian_behavior + obj.pedestrian_behavior,
            pedestrian_age=self.pedestrian_age + obj.pedestrian_age,
        )

    def to_dict(self) -> dict[str, int]:
        result = {}
        for value, count in self.object_motion.items():
            result[f"object_motion.{value}"] = count
        for value, count in self.rider_status.items():
            result[f"rider_status.{value}"] = count
        for value, count in self.pedestrian_behavior.items():
            result[f"pedestrian_behavior.{value}"] = count
        for value, count in self.pedestrian_age.items():
            result[f"pedestrian_age.{value}"] = count

        return result


def parse_args():
    parser = ArgumentParser(
        description="PandaSetのattributeの一意な値を出力します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output", type=Path, required=True, help="出力先")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def get_attribute_counter(sequence: Sequence) -> AttributeCounter:
    sequence.load_cuboids()
    # 先頭だけ見る

    result = AttributeCounter()
    for df in sequence.cuboids.data:
        counter = AttributeCounter(
            object_motion=Counter(df["attributes.object_motion"])
            if "attributes.object_motion" in df.columns
            else Counter(),
            rider_status=Counter(df["attributes.rider_status"])
            if "attributes.rider_status" in df.columns
            else Counter(),
            pedestrian_behavior=Counter(df["attributes.pedestrian_behavior"])
            if "attributes.pedestrian_behavior" in df.columns
            else Counter(),
            pedestrian_age=Counter(df["attributes.pedestrian_age"])
            if "attributes.pedestrian_age" in df.columns
            else Counter(),
        )
        result = result + counter
    return result


def get_df_from_sequence_counter(sequence_counter: dict[str, AttributeCounter]) -> pandas.DataFrame:
    row_list = []
    for sequence_id, attribute_counter in sequence_counter.items():
        tmp: dict[str, Any] = attribute_counter.to_dict()
        tmp["sequence_id"] = sequence_id
        row_list.append(tmp)

    return pandas.DataFrame(row_list)


def main() -> None:
    args = parse_args()
    set_default_logger()

    input_dir: Path = args.input_dir

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    sequence_counter: dict[str, AttributeCounter] = {}
    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のcuboidを読み込みます。")
        try:
            sequence_counter[sequence_id] = get_attribute_counter(sequence)
        except Exception:
            logger.warning(f"{sequence_id=}のcuboidの読み込みに失敗しました。", exc_info=True)
        finally:
            dataset.unload(sequence_id)

    df = get_df_from_sequence_counter(sequence_counter)
    output: Path = args.output
    output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(str(output), index=False)


if __name__ == "__main__":
    main()
