# ESM-VOTAFM
Implementation of "Efficient Siamese Model for Visual Object Tracking with Attention-Based Fusion Modules" on Pytorch. 

#  Environment setup
```bash
pip install -r requirements.txt
```

# Compile
cd /path/to/yourproject

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
Download the datasets：[GOT-10k](http://got-10k.aitestunion.com/downloads)

For detailed environmental configuration and cutting of datasets , please refer to [pysot](https://github.com/STVIR/pysot) or [SiamTrackers
](https://github.com/HonglinChu/SiamTrackers).

You can simply run the `my_train.py` file to initiate the process.



# Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot), [SiamTrackers
](https://github.com/HonglinChu/SiamTrackers) and [DROL](https://github.com/shallowtoil/DROL). 

We would like to express our sincere thanks to the contributors.
