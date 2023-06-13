import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import pandas
from pandaset import DataSet
from pandaset.sequence import Sequence

from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="PandaSetのcuboidのlabel数を出力します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output", type=Path, required=True, help="出力先")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def get_label_counter(sequence: Sequence) -> dict[str, int]:
    sequence.load_cuboids()
    # 先頭だけ見る
    df = sequence.semseg.data[0]
    tmp = Counter(df["class"])
    classes = sequence.semseg.classes
    return {classes[str(class_id)]: count for class_id, count in tmp.items()}


def main() -> None:
    args = parse_args()
    set_default_logger()

    input_dir: Path = args.input_dir

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    data: list[dict[str, Any]] = []
    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のsemsegのlabel一覧を取得します。")
        try:
            sequence.load_semseg()
            tmp = get_label_counter(sequence)
            tmp["sequence_id"] = sequence_id
            data.append(tmp)

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
