#! /usr/bin/sh

encounter_folder_local = ./Encounter
cd $encounter_folder_local
# unpack transfered reports .zip, ensuring old folder is removed \
rm -rf reports && unzip -o reports.zip
