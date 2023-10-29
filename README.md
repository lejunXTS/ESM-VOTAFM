# ESM-VOTAFM
Implementation of "Efficient Siamese Model for Visual Object Tracking with Attention-Based Fusion Modules" on Pytorch. 


cd /home/xxx/xxx/SiamTrackers/3-SiamRPN/SiamRPNpp-UP 

python setup.py build_ext --inplace

# Test tracker
You can either directly run the `my_test.py` file or use the following command to execute the script:

```bash
./bin/cmd_test.sh
```

# Eval tracker
You can either directly run the `my_eval.py` file or use the following command to execute the script:

```bash
./bin/cmd_eval.sh
```


# Training
For environment configuration, please refer to [pysot](https://github.com/STVIR/pysot) or [SiamTrackers
](https://github.com/HonglinChu/SiamTrackers).

You can simply run the `my_train.py` file to initiate the process.



# Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot), [SiamTrackers
](https://github.com/HonglinChu/SiamTrackers) and [DROL](https://github.com/shallowtoil/DROL). 

We would like to express our sincere thanks to the contributors.
