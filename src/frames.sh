#!/bin/bash

python src/feat_extract.py --data-list data/video.txt --model i3d_resnet50_v1_kinetics400 --save-dir ./features