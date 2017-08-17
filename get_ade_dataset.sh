#!/bin/bash

name=./ADEChallengeData2016

if [ ! -f ${name}.zip ]
then
	echo 'Downloading zip...'
	wget 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
fi

if md5sum --status -c adechallenge.md5; then
	echo 'MD5 matches'
else
	echo 'MD5 does not match'
	exit 1
fi

rm -rf ${name}/

echo 'Unzipping...'
unzip -q ${name}.zip

echo 'Adjusting directories and file locations...'

mkdir -p ${name}/train/images/
mkdir -p ${name}/train/annotations/
mkdir -p ${name}/val/images/
mkdir -p ${name}/val/annotations/

mv ${name}/images/training/* ${name}/train/images/
mv ${name}/images/validation/* ${name}/val/images/

mv ${name}/annotations/training/* ${name}/train/annotations/
mv ${name}/annotations/validation/* ${name}/val/annotations/

rm -rf ${name}/images
rm -rf ${name}/annotations

echo 'Done.'
