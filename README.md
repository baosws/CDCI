# Conditional Divergence based Causal Inference

This is the implementation of our paper: Bao Duong and Thin Nguyen. [Bivariate causal discovery via conditional divergence](https://openreview.net/forum?id=8X6cWIvY_2v). In First Conference on Causal Learning and Reasoning, 2022.

## Dependencies

- numpy
- pandas
- scikit-learn
- CausalDiscoveryToolbox (https://github.com/FenTechSolutions/CausalDiscoveryToolbox)

Step-by-step installation:
```
conda create -n CDCI -c conda-forge python=3 numpy pandas scikit-learn
conda activate CDCI
pip install cdt
```

## Data

Data sources:

- Goudet, Olivier, 2017, "Causality pairwise inference datasets. Replication Data for: "Learning Functional Causal Models with Generative Neural Networks"", https://doi.org/10.7910/DVN/3757KX, Harvard Dataverse, V1, UNF:6:GI+fYdz/lRcUp/ir2EoVbw== [fileUNF]
- DREAM4 challenge: https://www.synapse.org/#!Synapse:syn3049712/wiki/74628

Data is prepared in the following structure:
- Location: "data/{dataset name}.csv". Data set names should be CE-Cha, CE-Net, CE-Gauss, CE-Multi, D4S1, D4S2A, D4S2B, or D4S2C.
- Each data set should have at least 4 columns: SampleID, A, B, and Target.

## Usage:

```
python CDCI.py [-h] [--methods METHODS [METHODS ...]] [--datasets DATASETS [DATASETS ...]] [--maxdev MAXDEV] [--out PATH] [--hyper]

optional arguments:
  -h, --help            show this help message and exit
  --methods METHODS [METHODS ...], -m METHODS [METHODS ...]
                        Applicable methods: ANM, CDS, IGCI, RECI, CCS, CHD, CKL, CKM, CTV. [Default: All methods]
  --datasets DATASETS [DATASETS ...], -d DATASETS [DATASETS ...]
                        Available data sets: CE-Cha, CE-Gauss, CE-Net, CE-Multi, D4S1, D4S2A, D4S2B, and D4S2C. [Default: All data sets]
  --maxdev MAXDEV       Discretization parameter
  --out PATH, -o PATH   Output file
  --hyper               Test hyperparameters
```

Examples:

- Run CDCI methods on CE-Net and CE-Cha datasets:
```
python CDCI.py -m CCS CHD CKL CKM CTV -d CE-Net CE-Cha
```
- Evaluate hyperparameters for CDCI on real datasets:
```
python CDCI.py -m CCS CHD CKL CKM CTV -d D4S1 D4S2A D4S2B D4S2C --hyper
```

## Citation

If this code helps in your work, please consider citing us as:
```
@inproceedings{duong2022bivariate,
  title={Bivariate causal discovery via conditional divergence},
  author={Duong, Bao and Nguyen, Thin},
  booktitle={Conference on Causal Learning and Reasoning},
  pages={236--252},
  year={2022},
  organization={PMLR}
}

```
