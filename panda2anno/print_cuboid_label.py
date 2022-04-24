import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas
from pandaset import DataSet
from pandaset.sequence import Sequence

from panda2anno.common.annofab import get_label_id_from_pandaset
from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="PandaSetのcuboidのlabelの一覧を出力します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output", type=Path, required=True, help="出力先")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def get_unique_labels(sequence: Sequence) -> set[str]:
    sequence.load_cuboids()
    # 先頭だけ見る
    df = sequence.cuboids.data[0]

    result = set(df["label"].unique())
    logger.debug(f"{result=}")
    return result


def main() -> None:
    args = parse_args()
    set_default_logger()

    input_dir: Path = args.input_dir

    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    labels: set[str] = set()
    for sequence_id in sequence_id_list:
        sequence = dataset[sequence_id]
        logger.info(f"{sequence_id=}のcuboidのlabel一覧を取得します。")
        try:
            tmp_labels = get_unique_labels(sequence)
            labels = labels | tmp_labels
        except Exception:
            logger.warning(f"{sequence_id=}のcuboidをAnnofabのアノテーションへの変換に失敗しました。", exc_info=True)
        finally:
            dataset.unload(sequence_id)

    label_list = sorted(labels)
    label_ids = [get_label_id_from_pandaset(label) for label in label_list]

    df = pandas.DataFrame({"label_id": label_ids, "label_name": label_list})
    output: Path = args.output
    output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(str(output), index=False)


if __name__ == "__main__":
    main()
