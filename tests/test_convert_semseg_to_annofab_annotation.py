from panda2anno.convert_semseg_to_annofab_annotation import Semseg2Annofab
from pandaset import DataSet
import os
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

output_dir = Path("tests/out")
dataset = DataSet("tests/resources/pandaset/")

sequence_id = "001"
sequence = dataset[sequence_id]


def test_main():
    main_obj = Semseg2Annofab()

    main_obj.write_semseg_annotations(
        sequence, output_dir=output_dir / f"semseg/{sequence_id}", sequence_id=sequence_id
    )


def teardown_module(moduloe):
    dataset.unload(sequence_id)
