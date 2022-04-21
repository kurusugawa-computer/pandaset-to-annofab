import re


def get_annofab_label_id_from_pandaset(label: str) -> str:
    """pandasetのlabelからannofabのlabel_idを取得する"""
    # IDに使えない文字は"__"で置換する
    return re.sub("[^0-9A-Za-z-_.]", "__", label)
