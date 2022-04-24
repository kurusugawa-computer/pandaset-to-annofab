from panda2anno.convert_data_to_kitti import Pandaset2Kitti
from pandaset import DataSet
import os
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

output_dir = Path("tests/out")
dataset = DataSet("tests/resources/pandaset/")

sequence_id = "001"
sequence = dataset[sequence_id]


def test_main():
    main_obj = Pandaset2Kitti()

    main_obj.write_kitti_scene(sequence, output_dir=output_dir / f"kitti/{sequence_id}", sequence_id=sequence_id)

    dataset.unload(sequence_id)


def teardown_module(moduloe):
    dataset.unload(sequence_id)
