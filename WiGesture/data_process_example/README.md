# How to Run

1. Execute `process1.py` to convert the `csv` file into a `pkl` file.

2. Run one of the `process2` scripts based on your requirements:

    

   **Note:** We use `-1000` to represent the position of package loss. You can apply various interpolation methods to fill these gaps. We highly encourage you to try our CSI-BERT model to recover the lost packages. ([CSI-BERT](https://github.com/RS2002/CSI-BERT), [CSI-BERT2](https://github.com/RS2002/CSI-BERT2))

    

   (1) If you want to process each record into a long sequence, run `process2.py`. You can refer to `dataset.py` in [CSI-BERT2](https://github.com/RS2002/CSI-BERT2) for guidance.

    

   (2) If you prefer to split each record into multiple fixed-length samples, run `process2-split.py` and modify the `length` parameter in the code to your desired length. You can refer to `dataset.py` in [CSI-BERT](https://github.com/RS2002/CSI-BERT), [CrossFi](https://github.com/RS2002/CrossFi), [KNN-MMD](https://github.com/RS2002/CrossFi), and [LoFi](https://github.com/RS2002/LoFi/tree/main/network_examples) for usage instructions.

    

   (3) `process2-squeeze-split.py` functions similarly to `process2-split.py`, but it excludes all lost packages.