#!/bin/bash
export PYTHONPATH=/path/to/project:$PYTHONPATH   #修改地址
export PYTHONPATH=/path/to/siamrpnpp:$PYTHONPATH   #修改地址
export PYTHONPATH=/path/to/toolkit:$PYTHONPATH   #修改地址

python ./bin/my_eval.py \
--tracker_path ./results \
--dataset OTB100 \
--num 8 \
--tracker_name  'checkpoin*' 
 
