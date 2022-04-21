import pandas
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import subprocess


def parse_args():
    parser = ArgumentParser(
        description="`anno3d`コマンドを使って、Annofabのアノテーション仕様にcuboidを追加します。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--project_id", type=str, required=True, help="Annofabのproject_id")
    parser.add_argument("--label_csv", type=Path, required=True, help="label_idとlabel_nameのCSV")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_label = pandas.read_csv(str(args.label_csv))

    for _, row in df_label.iterrows():
        label_id = row["label_id"]
        label_name = row["label_name"]

        command = [
            "anno3d",
            "project",
            "put_cuboid_label",
            "--project_id",
            args.project_id,
            "--label_id",
            label_id,
            "--ja_name",
            label_name,
            "--en_name",
            label_name,
            "--color",
            "(255,0,0)",
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
