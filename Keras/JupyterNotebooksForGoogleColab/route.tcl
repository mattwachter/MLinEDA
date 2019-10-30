# Route design that has already been placed, e.g. with Eh?Placer
# (https://www.ucalgary.ca/karimpour/node/10) and imported into Encounter with
# import_save.tcl

source design_placed.enc
# Base Encounter is licensed for up to 8 threads
setMultiCpuUsage -localCpu 8
# Routing should not be timing driven since ISPD2015 has no timing constraints.
setNanoRouteMode -drouteAutoStop false
globalRoute
detailRoute
# TODO Create map file (specific to each layouts layers)
oasisOut design_routed.oas -mapFile ./oasisOut_mgc_fft_1.map -libName DesignLib \
    -dieAreaAsBoundary
# Specify file name to be saved
saveDesign design_routed.enc