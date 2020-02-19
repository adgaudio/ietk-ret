#!/usr/bin/env bash

# Small script to copy healthy images into a dedicated directory.  Note: using
# hard links to save disk space (I don't actually copy the files).  Only works
# on linux.
set -e
set -u

for fp in data/messidor/*csv ; do
  base_num="$(echo $fp | sed -r 's/.*Annotation_(Base..)\.csv$/\1/')"
  echo $base_num
  mkdir -p data/messidor_healthy/$base_num
  cat $fp | cut -f1,3 -d, | grep ,0|cut -f1 -d, | parallel cp -al data/messidor/$base_num/{} data/messidor_healthy/$base_num/{}
  mkdir -p data/messidor_grade1/$base_num
  cat $fp | cut -f1,3 -d, | grep ,1|cut -f1 -d, | parallel cp -al data/messidor/$base_num/{} data/messidor_grade1/$base_num/{}
  mkdir -p data/messidor_grade2/$base_num
  cat $fp | cut -f1,3 -d, | grep ,2|cut -f1 -d, | parallel cp -al data/messidor/$base_num/{} data/messidor_grade2/$base_num/{}
  mkdir -p data/messidor_grade3/$base_num
  cat $fp | cut -f1,3 -d, | grep ,3|cut -f1 -d, | parallel cp -al data/messidor/$base_num/{} data/messidor_grade3/$base_num/{}
done
