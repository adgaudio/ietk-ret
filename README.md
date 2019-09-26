# Illumination Correction

This code implements an illumination correction filter by making use of
the Dark Channel Prior dehazing theory for removing fog from images of outdoors.
I apply it to retinal images from the Messidor dataset.


# Pre-requisite python libraries
Required libraries to run the illumination correction and tests, and
to generate an illumination corrected dataset.
You may need most recent version of opencv

```
cv2
glob
joblib
matplotlib
multiprocessing
numpy
os
pandas
random
scipy
seaborn
```

# Background papers:

Saved in pdf files here:

./dark_channel.pdf
./guided_filter.pdf


# Run the scripts

NOTE: I included a subset of the Messidor dataset of retinal fundus
images in the SVN repository so these commands will work properly.

#### Run the illumination correction pipeline just to check that it works
```
python dehaze.py
```

#### Run the dehaze tests to visually evaluate the results of applying the pipeline

Note: These tests may use up all your ram until OOM, but shouldn't
cause the computer to freeze (at least not on Arch linux).  I do that to
queue up plots so the plots render more quickly.

FYI: in the script, there are 3 tests.  Only one should be enabled:
```
    test1 = False
    test2 = False
    test3 = True
```
```
python dehaze_testing.py
```

#### To train the CNN, I used this script to modify my version of the Messidor dataset:

```
./create_dehazed_dataset
```
I then passed that into a pytorch CNN library I developed for my
research.  I didn't share the library here, but I can share if you need
to see it.

This command requires a messidor dataset in ./data/messidor.  I include
a small sample of Messidor there, but to reproduce results shown in
slides, the CNN needs the full messidor dataset.


#### To evaluate whether amount of shadow correlates to the grade, I run the following script to create plots:

I added a couple comments in the file here, since I hadn't completed
this work during my presentation.

The script by default uses results saved in a csv file that I computed
on the full dataset.

```
python quantify_shadows.py
```
