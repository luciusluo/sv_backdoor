#!/bin/bash
# add label column
# voxceleb1_iden_ori.txt --> voxceleb1_iden.txt

file='./voxceleb1_iden_ori.txt'
label_file='./speaker_id.txt'

awk 'NR==FNR{split($2,split_,"/"); spk=split_[1]; line2spk[$0]=spk}\
  NR!=FNR{spk2label[$1]=$2}\
  END{for (line in line2spk) {
        spk=line2spk[line];
        label=spk2label[spk];
        print line" "label
        }
     }' ${file} ${label_file} | sort -n -k3 > ./voxceleb1_iden.txt
