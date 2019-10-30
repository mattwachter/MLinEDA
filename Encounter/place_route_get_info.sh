# These commands are meant to be executed separately in the folder of each
# benchmark
# The respective tools (Eh?Placer) and scripts (from Encounter/) should be
# placed in the users home directory ~/bin

design_path=$HOME/ispd_2015_contest_benchmark/mgc_fft_1_encounter/

cd $design_path
# Source PATH of EDA tools. Depends on local setup of Encounter installation

# Batch process tcl script with Encounter Design Implementation system.
encounter -no_gui -init design_placed.enc -batch -file get_info.tcl
# Clean out old zip file and create new one to facilitate data transfer.
rm reports.zip && zip -r reports reports

# nohup runs the command in the background and does not stop even when the ssh
# session is interrupted
nohup ~/bin/EhPlacer_ispd01.placer -tech_lef ./tech.lef -cell_lef ./cells.lef -floorplan_def ./floorplan.def -output ./EhPlacerOutput.def -placement_constraints ./placement.constraints -verilog ./design.v -cpu 8 >EhPlacer.log 2>&1 & # runs in background, doesn't create nohup.out

nohup sh ~/bin/EhPlacerCMD.sh
s_tools
encounter -no_gui -batch -file ~/bin/import_save.tcl
nohup encounter -no_gui -batch -file ~/bin/route.tcl >/dev/null 2>&1 & # runs in background, doesn't create nohup.out

nohup encounter -no_gui -batch -file ~/bin/get_info.tcl >/dev/null 2>&1 & # runs in background, doesn't create nohup.out

# Afterwards the generated files can be processed with the Keras/extract_data.py
# script