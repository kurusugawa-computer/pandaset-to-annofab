#!/bin/bash -eux

# anno3dコマンドで、入力データ/補助情報をAnnofabに登録します。
PROJECT_ID=$1
KITTI_DIR=$2

# プロジェクトトップに移動する
SCRIPT_DIR=$(cd "$(dirname $0)"; pwd)
pushd "${SCRIPT_DIR}/../"

PROJECT_ID=a125646c-f37d-4b49-9de1-ef04ea25df14
while read -r f; do

  # ファイル一つ毎の処理
  echo "file: $f"
  anno3d project upload_scene --project_id ${PROJECT_ID} --force --upload_kind data --sensor_height 0 --scene_path $f 

done < <(find ${KITTI_DIR} -mindepth 1 -maxdepth 1)


popd