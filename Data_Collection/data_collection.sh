#!/bin/bash
# input format is 
# ./data_collection.sh {username} {eventname} {eventnumber} {timeinterval} {node}


varpath=./$1/$2$3
if [ ! -d $varpath  ];then
  mkdir -p $varpath
fi
python event_input.py -u $1 -e $2 -n $3 -t $4 -c $5 & python camerarecord.py -u $1 -e $2 -n $3 -t $4 -c $5 & arecord --device=hw:2,0 -d $4 -r 48000 -c 1 -f S16_LE $varpath/audio.wav
wait
echo Data collected.
var=node$5_$(date +%y%m%d%H%M)_p$1_$2$3
zip -q -r $var.zip $varpath
echo Packed

