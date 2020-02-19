#!/usr/bin/env bash

python bin/separability/eval_separability.py
python bin/qualdr/qualdr_analysis.py
# not used -- python bin/qualdr/qualdr_analysis_v2.py

# correlation plots, the tables about idrid, rite and qualdr.
# this is the most useful code here.
python bin/segmentation/joint_analysis.py

# RITE vessel segmentation maps
python bin/paper_plot_segmentation_maps2.py

# Qualitative visualization of pre-processing methods
python ./bin/paper_plot_qualitative.py

# python ./bin/paper_table_qualdr_dataset_class_distribution.py
# paper_table_idrid_dataset_class_distribution.py
python ./bin/paper_plot_fig1.py
python ./bin/paper_plot_fig3.py
./bin/paper_other_plots.sh

