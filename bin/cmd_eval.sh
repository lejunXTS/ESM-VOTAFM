#!/bin/bash
export PYTHONPATH=/path/to/project:$PYTHONPATH   #Change it to your address.
export PYTHONPATH=/path/to/siamrpnpp:$PYTHONPATH   #Change it to your address.
export PYTHONPATH=/path/to/toolkit:$PYTHONPATH   #Change it to your address.

python ./bin/my_eval.py \
--tracker_path ./results \
--dataset OTB100 \
--num 8 \
--tracker_name  'checkpoin*' 
 
