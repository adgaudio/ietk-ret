Code accompanying the Pixel Color Amplification paper, 

> A. Gaudio, A. Smailagic, A. Campilho, “Enhancement of Retinal Fundus Images via Pixel Color Amplification,” In: Karray F., Campilho A., Wang Z. (eds) Image Analysis and Recognition. ICIAR 2020. Lecture Notes in Computer Science, vol 11663. Springer, Cham  (accepted)

(**to be updated upon formal publication by Springer**):  The final authenticated publication is available online at https://doi.org/[insert DOI]

- [Paper, accepted for publication after review (14 pages, some updates)](./gaudio_iciar2020_after_reviewers.pdf)
- [Paper, before reviewers (20 pages, extra experiments and analysis)](./gaudio_iciar2020_after_reviewers.pdf)

- [slides, short version](https://docs.google.com/presentation/d/1Xp7mvulf-UB2R-jc2Z6K7DCuJXBEpxaiVJ_ae11nVmE/edit?usp=sharing)
- [slides, long version (**forthcoming**)]() Used for my PhD Qualifier exam.
I will share them after I publish additional results they contain.


# Reproducing Results

The below steps should completely reproduce results for the paper.

I tried to leave this in its "raw" state so the code is as close as
possible to the state it was in when I generated results.  This means it
isn't cleaned up in some places.  I did move many
files around since generating results.  I also released a
[simplepytorch](https://github.com/adgaudio/simplepytorch) library in order
to make this reproducible.

There are three git tags identifying where the work was done.

- `iciar2020_not_reproducible`  -  Deep network results were obtained the work using a private library that I have subsequently open sourced.
- `iciar2020`  -  I attempted to make the results completely reproducible.
- `iciar2020_v2`  -  Some code changes to address reviewer comments.  Note
that the paper before reviewers was 20 pages, and I was asked to cut it to 14
to length requirements.  Thus, several results obtained are not published.

```
check out the git tag to reproduce results
$ git checkout iciar2020_v2
```

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
# QualDR results (not used in published version of paper)
cat ./bin/qualdr/reproduce_qualdr_grading.sh
```
```
# RITE and IDRiD results  (RITE not used in the published version of paper)
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
