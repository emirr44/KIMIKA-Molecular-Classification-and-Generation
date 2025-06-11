# Dataset

Both classification and molecule generation codes in this project rely on same dataset, bace.csv. The BACE dataset focuses on inhibitors of human beta-secretase 1 (BACE-1). It includes both quantitative (IC50 values) and qualitative (binary labels) binding results.

'''wget -O bace.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
mkdir -p ./molecule_datasets/bace/raw
mv bace.csv ./molecule_datasets/bace/raw/bace.csv
'''

Since mentioned dataset is realtively small (~1513 molecules)
