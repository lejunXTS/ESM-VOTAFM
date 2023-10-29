#!/bin/bash
export PYTHONPATH=/path/to/project:$PYTHONPATH   #修改地址
export PYTHONPATH=/path/to/siamrpnpp:$PYTHONPATH   #修改地址
export PYTHONPATH=/path/to/toolkit:$PYTHONPATH   #修改地址

START=35
END=49
seq $START 1 $END | \
    xargs -I {} echo "./models/siamrpnpp_alexnet/snapshot/checkpoint_e{}.pth" | \
    xargs -I {} python -u ./bin/my_test.py --snapshot {} --config ./models/siamrpnpp_alexnet/config.yaml \
    --dataset OTB100 2>&1 | tee ./models/siamrpnpp_alexnet/logs/test_dataset.log

#START 和 END 为开始测试和结束测试的轮数
