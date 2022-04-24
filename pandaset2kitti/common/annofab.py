import re


def get_annofab_label_id_from_pandaset(label: str) -> str:
    """pandasetのlabelからannofabのlabel_idを取得する"""
    # IDに使えない文字は"__"で置換する
    return re.sub("[^0-9A-Za-z-_.]", "__", label)


def get_input_data_id_from_pandaset(sequence_id: str, frame_index: int) -> str:
    """
    pandasetのsequence_idとフレーム番号から、Annofabのinput_data_idを取得する。
    """
    return f"{sequence_id}-{str(frame_index)}"
