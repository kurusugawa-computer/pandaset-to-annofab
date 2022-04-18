import json
import logging
import math
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def set_default_logger():
    """
    デフォルトのロガーを設定する。
    """

    logging_formatter = "%(levelname)-8s : %(asctime)s : %(filename)s : %(name)s : %(funcName)s : %(message)s"
    logging.basicConfig(format=logging_formatter)
    logging.getLogger("dgp2kitti").setLevel(level=logging.DEBUG)
    logging.getLogger("__main__").setLevel(level=logging.DEBUG)


def quaterion_to_euler_angles(quaterion: List[float]) -> List[float]:
    """
    quaterion から　YXZのオイラー角を求める。以下のコードを移植したもの
    https://github.com/BabylonJS/Babylon.js/blob/40ded9ccf1e1bd8ac9cdf3a26909d3e12bc60ab8/src/Maths/math.vector.ts#L2970-L3001

    Args:
        quaterion: wxyzの1次元配列

    Returns:
        オイラー角[x,y,z]

    """
    qw = quaterion[0]
    qx = quaterion[1]
    qy = quaterion[2]
    qz = quaterion[3]

    sqx = qx * qx
    sqy = qy * qy
    sqz = qz * qz
    sqw = qw * qw

    zAxisY = qy * qz - qx * qw
    limit = 0.4999999

    if zAxisY < -limit:
        euler_y = 2 * math.atan2(qy, qw)
        euler_x = math.pi / 2
        euler_z = 0.0
    elif zAxisY > limit:
        euler_y = 2 * math.atan2(qy, qw)
        euler_x = -math.pi / 2
        euler_z = 0.0
    else:
        euler_z = math.atan2(2.0 * (qx * qy + qz * qw), (-sqz - sqx + sqy + sqw))
        euler_x = math.asin(-2.0 * (qz * qy - qx * qw))
        euler_y = math.atan2(2.0 * (qz * qx + qy * qw), (sqz - sqx - sqy + sqw))

    return [euler_x, euler_y, euler_z]


def euler_angles_to_quaterion(euler_angles: List[float]) -> List[float]:
    """
    YXZのオイラー角からクォータニオンを求める。

    以下のサイトから移植
    https://github.com/BabylonJS/Babylon.js/blob/40ded9ccf1e1bd8ac9cdf3a26909d3e12bc60ab8/src/Maths/math.vector.ts#L3259-L3275
    https://github.com/BabylonJS/Babylon.js/blob/40ded9ccf1e1bd8ac9cdf3a26909d3e12bc60ab8/src/Maths/math.vector.ts#L3198-L3201

    Args:
        euler_angles: YXZのオイラー角[x,y,z]

    Returns:
        クォータニオン[w,x,y,z]
    """
    yaw = euler_angles[1]
    pitch = euler_angles[0]
    roll = euler_angles[2]

    halfRoll = roll * 0.5
    halfPitch = pitch * 0.5
    halfYaw = yaw * 0.5

    sinRoll = math.sin(halfRoll)
    cosRoll = math.cos(halfRoll)
    sinPitch = math.sin(halfPitch)
    cosPitch = math.cos(halfPitch)
    sinYaw = math.sin(halfYaw)
    cosYaw = math.cos(halfYaw)

    qx = (cosYaw * sinPitch * cosRoll) + (sinYaw * cosPitch * sinRoll)
    qy = (sinYaw * cosPitch * cosRoll) - (cosYaw * sinPitch * sinRoll)
    qz = (cosYaw * cosPitch * sinRoll) - (sinYaw * sinPitch * cosRoll)
    qw = (cosYaw * cosPitch * cosRoll) + (sinYaw * sinPitch * sinRoll)
    return [qw, qx, qy, qz]


def get_hash_code(value: str) -> int:
    """
    文字列からuint32の整数に変換する。

    Args:
        value: 文字列

    Returns:
        uint32の値
    """
    hash_value = 7
    for c in value:
        # 32 bit integer
        hash_value = (31 * hash_value + ord(c)) & 4294967295
    return hash_value


def read_lines(filepath: str) -> List[str]:
    """ファイルを行単位で読み込む。改行コードを除く"""
    with open(filepath) as f:
        lines = f.readlines()
    return [e.rstrip("\r\n") for e in lines]


def read_lines_except_blank_line(filepath: str) -> List[str]:
    """ファイルを行単位で読み込む。ただし、改行コード、空行を除く"""
    lines = read_lines(filepath)
    return [line for line in lines if line != ""]


def _get_file_scheme_path(str_value: str) -> Optional[str]:
    """
    file schemaのパスを取得する。file schemeでない場合は、Noneを返す

    """
    FILE_SCHEME_PREFIX = "file://"
    if str_value.startswith(FILE_SCHEME_PREFIX):
        return str_value[len(FILE_SCHEME_PREFIX) :]
    else:
        return None


def get_list_from_args(str_list: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    文字列のListのサイズが1で、プレフィックスが`file://`ならば、ファイルパスとしてファイルを読み込み、行をListとして返す。
    そうでなければ、引数の値をそのまま返す。
    ただしNoneの場合はNoneを返す。

    Args:
        str_list: コマンドライン引数で指定されたリスト、またはfileスキームのURL

    Returns:
        コマンドライン引数で指定されたリスト。
    """
    if str_list is None or len(str_list) == 0:
        return None

    if len(str_list) > 1:
        return str_list

    str_value = str_list[0]
    path = _get_file_scheme_path(str_value)
    if path is not None:
        return read_lines_except_blank_line(path)
    else:
        return str_list


def get_json_from_args(target: Optional[str] = None) -> Any:
    """
    JSON形式をPythonオブジェクトに変換する。
    プレフィックスが`file://`ならば、ファイルパスとしてファイルを読み込み、Pythonオブジェクトを返す。
    """

    if target is None:
        return None

    path = _get_file_scheme_path(target)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    else:
        return json.loads(target)
