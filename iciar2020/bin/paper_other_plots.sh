  # ./data/histograms_idrid_plots/separability_consistency/topN-separability_mean_of_channels.png \
  # ./data/histograms_idrid_plots/separability_consistency/top15-separability.tex \
cp \
  ./data/plots/brighten_darken \
  ./data/plots/sharpen_fundus.png \
  ./data/plots/rite_segmentation.png \
  ./data/plots/qualdr/* \
  ./data/plots/joint_analysis/* \
  ./paper/figures
function cpq() {
  local idx=$1
  cp ./data/plots/qualitative/qualitative-$idx-* \
    ./paper/figures/qualitative-$idx.png
}
cpq 344
cpq 416
cpq 1
cpq 682
cpq 56


