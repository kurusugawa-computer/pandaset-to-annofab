import logging
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from pandaset import DataSet

from panda2anno.common.utils import set_default_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description="指定したカメラ画像の`00.jpg`を別のディレクトリにコピーします。コピー後のファイル名は`{sequence_id}__{camera}_00.jpg`です。",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="pandasetのディレクトリ")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="出力先ディレクトリ")
    parser.add_argument("--camera", type=str, required=True, help="出力対象のcamera")

    parser.add_argument("--sequence_id", type=str, nargs="+", required=False, help="出力対象のsequence id")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_default_logger()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    dataset = DataSet(str(input_dir))

    if args.sequence_id is None:
        sequence_id_list = dataset.sequences()
    else:
        sequence_id_list = args.sequence_id

    camera: str = args.camera

    output_dir.mkdir(exist_ok=True, parents=True)

    success_count = 0
    for sequence_id in sequence_id_list:
        camera_dir = input_dir / sequence_id / "camera" / camera
        original_first_frame_filename = "00.jpg"
        original_image_file = camera_dir / original_first_frame_filename
        if not original_image_file.exists():
            logger.warning(f"{original_image_file}は存在しません。ファイルのコピーをスキップします。")
            continue

        output_file = output_dir / f"{sequence_id}__{camera}__00.jpg"

        try:
            shutil.copy(original_image_file, output_file)
            success_count += 1
        except Exception:
            logger.warning(f"{original_image_file}のファイルコピーに失敗しました。", exc_info=True)

    logger.info(f"{success_count} 件の{camera}の画像ファイルを{output_dir}にコピーしました。")


if __name__ == "__main__":
    main()
