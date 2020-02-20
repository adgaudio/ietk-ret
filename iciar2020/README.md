Code accompanying the Pixel Color Amplification paper, 

TODO: paper citation.


# Reproducing Results

I tried to leave this in its "raw" state so the code is as close as
possible to the state it was in when I generated results.  This means it
isn't cleaned up in some places.  I did move many
files around since generating results.  I also released a
[simplepytorch](https://github.com/adgaudio/simplepytorch) library in order
to make this reproducible.

```
# current working directory should be same as this README.md
```

```
# download and unzip the datasets into ./data.  Make a directory structure like this:
 $ mkdir data
 $ ls data/{IDRiD_segmentation,RITE,arsn_qualdr}
data/arsn_qualdr

data/IDRiD_segmentation:
'1. Original Images'  '2. All Segmentation Groundtruths'   CC-BY-4.0.txt   LICENSE.txt

data/RITE:
AV_groundTruth.zip  introduction.txt  read_me.txt  test  training
```

```
# QualDR results (assuming you have access to the private dataset
cat ./bin/qualdr/reproduce_qualdr_grading.sh
```
```
# RITE and IDRiD results
cat ./bin/segmentation/reproduce.py
```
```
# Separability on IDRiD train (This goes exhaustively through all models, all images, all label types, which is probably unnecessary).
python ./bin/separability/gen_histograms.py
```
```
# plots and tables
./bin/all_analysis.sh
```
