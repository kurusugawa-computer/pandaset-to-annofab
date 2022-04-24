# pandaset-to-kitti
[Pandaset](https://scale.com/open-datasets/pandaset)をAnnofabに登録するためのスクリプトです。



# Requriements
* Python3.9+
* annofabcli
* anno3d（社内PyPI）
* [pandaset-devkit](https://github.com/scaleapi/pandaset-devkit)


# Usage

## 拡張KITTI形式に変換する


```
# sequence_idが001で、10フレームごとにKITTIに変換する
$ poetry run python -m panda2anno.convert_data_to_kitti --input_dir pandaset_dir --output out/kitti \
 --sequence_id 001 --sampling_step 10


$ tree out/kitti/001
out/kitti/001
├── calib-front_camera
│   ├── 0.txt
│   ├── 001-0.txt
│   ├── 001-10.txt
│   ├── 001-20.txt
│   ├── 001-30.txt
│   ├── 001-40.txt
│   ├── 001-50.txt
│   ├── 001-60.txt
│   └── 001-70.txt
├── image-front_camera
│   ├── 0.jpg
│   ├── 001-0.jpg
│   ├── 001-10.jpg
│   ├── 001-20.jpg
│   ├── 001-30.jpg
│   ├── 001-40.jpg
│   ├── 001-50.jpg
│   ├── 001-60.jpg
│   └── 001-70.jpg
│
...
│
├── scene.meta
└── velodyne
    ├── 0.bin
    ├── 001-0.bin
    ├── 001-10.bin
    ├── 001-20.bin
    ├── 001-30.bin
    ├── 001-40.bin
    ├── 001-50.bin
    ├── 001-60.bin
    └── 001-70.bin

13 directories, 110 files
```

anno3dコマンドでデータの登録

```
$ anno3d project upload_scene --project_id ${PROJECT_ID} --force --upload_kind data \
 --sensor_height 0 --scene_path out/kitti/001 
```



## cuboidをAnnofabのSimpleアノテーションに変換する

```
# sequence_idが001のcuboidsを、10フレームごとにAnnofabのSimpleアノテーションに変換する
$ poetry run python -m panda2anno.convert_cuboid_to_annofab_annotation.py --input_dir pandaset_dir --output out/cuboids \
 --sequence_id 001 --sampling_step 10


$ tree out/cuboids
out/cuboids
├── 001
│   ├── 001-0.json
│   ├── 001-10.json
│   ├── 001-20.json
│   ├── 001-30.json
│   ├── 001-40.json
│   ├── 001-50.json
│   ├── 001-60.json
│   └── 001-70.json

$ cat out/cuboids/001/001-0.json | jq

{
    "details": [
        {
            "annotation_id": "2c4dbdea-845e-4d29-8a94-9c86feb536fe",
            "label": "Car",
            "attributes": {
                "object_motion": "Stopped",
                "rider_status": "",
                "pedestrian_behavior": "",
                "pedestrian_age": ""
            },
            "data": {
                "data": "{\"shape\":{\"dimensions\":{\"width\":1.867,\"height\":1.673,\"depth\":4.629},\"location\":{\"x\":-7.878498533504047,\"y\":36.81295918368881,\"z\":0.20847061576417758},\"rotation\":{\"x\":0,\"y\":0,\"z\":4.745545544005255},\"direction\":{\"front\":{\"x\":0.033150488800512974,\"y\":-0.9994503715003997,\"z\":0.0},\"up\":{\"x\":0.0,\"y\":0.0,\"z\":1.0}}},\"kind\":\"CUBOID\",\"version\":\"2\"}",
                "_type": "Unknown"
            }
        }
    ]
}


$ annofabcli annotation import --project_id ${PROJECT_ID} --annotation out/cuboids --task_id 001

```



## semantic segmentationをAnnofabのSimpleアノテーションに変換する

```
# sequence_idが001のsemantic segmentationを、10フレームごとにAnnofabのSimpleアノテーションに変換する
$ poetry run python -m panda2anno.convert_semseg_to_annofab_annotation.py --input_dir pandaset_dir --output out/semseg \
 --sequence_id 001 --sampling_step 10


$ tree out/semseg
out/semseg/001
├── 001-0
│   ├── 07833919-5540-46f2-af93-f713f966bca9
│   ├── 24fd8651-b61b-45f6-b7d2-431253e8a641
├── 001-0.json
├── 001-10
│   ├── 17a5400e-23cc-45bf-ae94-e73172561cfc
│   ├── 3226b98a-baed-4e0b-b99b-96b7555326bf
├── 001-10.json
...



$ cat out/semseg/001/001-0.json | jq
{
  "details": [
    {
      "annotation_id": "39e7a986-84d7-46b1-ab66-1f6f4d3f2bb6",
      "label": "Vegetation",
      "data": {
        "data": "001-0/39e7a986-84d7-46b1-ab66-1f6f4d3f2bb6",
        "_type": "Unknown"
      }
    }
  ]
}


$ cat out/semseg/001/001-0/07833919-5540-46f2-af93-f713f966bca9 | jq

{
  "points": [
    143446,
    143449,
    143452,
    143470,
    143473,
    143476
  ],
  "kind": "SEGMENT",
  "version": "1"
}



$ annofabcli annotation import --project_id ${PROJECT_ID} --annotation out/semseg --task_id 001

```
