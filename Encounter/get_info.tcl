# Run with
# encounter -no_gui -batch -file import_sae
# Saves information to checkDesign/fft.main.htm.ascii,
# should be run after placement and separately after routing


# Cohoose which design stages should be analyzed
set analyze_placed 1
set analyze_routed 1
# Choose size of square tile in um, 1000 dbu = 1um
set box_size 6


if {$analyze_placed} {
    # Clean out old reports
    file delete -force -- ./reports/placed
    # Create separate directories for each report type
    file mkdir ./reports/placed
    file mkdir ./reports/placed/verifyMetaldensity
    file mkdir ./reports/placed/verifyPowerVia
    file mkdir ./reports/placed/dbGet
}

if {$analyze_routed} {
    # Clean out old reports
    file delete -force -- ./reports/routed
    # Create separate directories for each report type
    file mkdir ./reports/routed
    file mkdir ./reports/routed/verifyMetaldensity
    file mkdir ./reports/routed/verifyPowerVia
    file mkdir ./reports/routed/dbGet
}

# ******************************** proc's **************************************


# ***************** Query information with dbQuery & dbGet commands ************
# Encounter Cmd ref p.4052 & p. 4039ff
# Saves various metrics from a rectangle box_cur to reports/dbGet/$box_name.csv
proc area_data_base_query_placed {box_cur box_name} {
    # Select all instances in current quadrant
    set insts_cur [dbQuery -area $box_cur -objType {inst}]
    # set markers_cur [dbQuery -area $box_cur -objType {marker}]
    # set nets_cur [dbQuery -area $box_cur -objType {net}]
    # Select all instance terminals (pins) in current quadrant
    set instTerms_cur [dbQuery -area $box_cur -objType {instTerm}]

    # Write CSV heading
    set csv_string ""
    # set csv_line_list "inst_name area numInstTerms"
    # set csv_string [csv_append_line $csv_string $csv_line_list]
    set csv_dataTypes "instTerms_pt_x instTerms_pt_y"
    puts "Recording the following data fields for each box: $csv_dataTypes"

    # Record list of x and y coordinates of pins
    # TODO Calculate standard deviation in python
    set csv_line_list "instTerms_pt_x [getInstTerms_pt_x $instTerms_cur]"
    set csv_string [csv_append_line $csv_string $csv_line_list]
    set csv_line_list "instTerms_pt_y [getInstTerms_pt_y $instTerms_cur]"
    set csv_string [csv_append_line $csv_string $csv_line_list]

    # Blank line before recording of instance specific information
    set csv_line_list "\n"
    set csv_string [csv_append_line $csv_string $csv_line_list]

    # Heading for instance names
    set csv_line_list "instName"
    foreach inst $insts_cur {
        set inst_cur_name [dbGet $inst.name]
        lappend csv_line_list "$inst_cur_name"
    }
    set csv_string [csv_append_line $csv_string $csv_line_list]

    # Area of each instance
    set csv_line_list "area"
    foreach inst $insts_cur {
        set inst_cur_area [dbGet $inst.area]
        lappend csv_line_list "$inst_cur_area"
    }
    set csv_string [csv_append_line $csv_string $csv_line_list]

    # Number of instance terminals (pins)
    set csv_line_list "numInstTerms"
   foreach inst $insts_cur {
        set numInstTerms [get_NumInstTerms $inst]
        lappend csv_line_list "$numInstTerms"
    }
    set csv_string [csv_append_line $csv_string $csv_line_list]

    # Write data to CSV file
    write_string $csv_string "reports/placed/dbGet/$box_name.csv"
}


proc area_data_base_query_routed {box_cur box_name} {
    set markers_cur [dbQuery -area $box_cur -objType {marker}]

    # write the subtypes of all markers of a tile into a csv
    set marker_subtypes ""
    for {set i 0} {$i < [llength $markers_cur]} {incr i} {
        lappend marker_subtypes [dbGet -i $i $markers_cur.subtype]
        }
    set csv_line_list "marker_subtype $marker_subtypes"

    # Write box-specific information to CSV file
    set csv_string ""
    set csv_string [csv_append_line $csv_string $csv_line_list]
    # Write data to CSV file
    write_string $csv_string "reports/routed/dbGet/$box_name.csv"
}


# ***************** Procedures instead of TCL packages *************************
# Use of Tcl CSV processing package, etc. is not possible since it is not
# supported by the Encounter Tcl interpreter.
# The values for each field are assumed to be strings or numbers like "12.34"
# The fields are separated by commas (",") and lines are finished with a
# UNIX style newline ("\n")

# Append a line of comma separated values from $csv_line_list to a $csv_string
# and return that $csv_string
proc csv_append_line {csv_string csv_line_list} {
    foreach item $csv_line_list {
        append csv_string $item ","
    }
    append csv_string "\n"
    return $csv_string
}


# Write string to file
proc write_string {string_output file_name} {
    # Open file with read and write privileges for the owner
    set fileId [open $file_name w]
    puts -nonewline $fileId "$string_output"
    close $fileId
    puts "Saved file $file_name to disk."
}


# ***************** Procedures instead of TCL packages *************************


# Add up all terminals of the input instances
proc get_NumInstTerms {insts_cur} {
    set numInstTerms 0
    foreach inst $insts_cur {
        incr numInstTerms [llength [dbGet $inst.instTerms]]
    }
    return $numInstTerms
}


# Check whether terminals of nets connected to $insts_cur are
# are within or out of $box_cur
proc get_NumLocalGlobalNets {box_cur} {
    set x1 [lindex $box_cur 0]
    set x2 [lindex $box_cur 1]
    set y1 [lindex $box_cur 2]
    set y2 [lindex $box_cur 3]
    set numLocalNets 0
    set numGlobalNets 0
    set instTerms_cur [dbQuery -area $box_cur -objType {instTerm}]
    puts "numInstTerms_cur: [llength $instTerms_cur]"
    # Get nets connected to at least one terminal in the current quadrant
    set nets_cur ""
    foreach term $instTerms_cur {
        set net_cur [dbGet $term.net]
        lappend nets_cur $net_cur
    }
    # puts "numNets_cur: [llength $nets_cur]"
    foreach net $nets_cur {
        set terms_connected [dbGet $net.allTerms]
        # puts "numTerms_connected: [llength $terms_connected]"
        set numLocalTerms 0
        set numGlobalTerms 0
        foreach term $terms_connected {
            # Check if coordinates of terminal are within $box_cur
            set pt_x [dbGet $term.pt_x]
            set pt_y [dbGet $term.pt_y]
            # puts "pt_x: $pt_x"
            # puts "pt_y: $pt_yj"
            if {(($pt_x >= $x1) && ($pt_x <= $x2)) &&
                (($pt_y >= $y1) && ($pt_y <= $y2))} {
                incr numLocalTerms
                } else {
                # Handle external terminal
                # If any Terminal is outside the box it cannot be a local net.
                incr numGlobalTerms
            }
                        # Criterion for local net from Tabrizi et al. (2018)
            if {($numLocalTerms >= 2) && ($numGlobalTerms == 0)} {incr numLocalNets}
            # All of the nets we are looking at necessarily have on terminal inside
            # the box.
            # Criterion for global nets from Tabrizi et al. is at least one local
            # terminal and at least one global terminal
            if {($numGlobalTerms >=1 ) && ($numLocalTerms >= 1)} {incr numGlobalNets}
        }
        # puts "numLocalTerms: $numLocalTerms, numGlobalTerms: $numGlobalTerms"
    }
    puts "numLocalNets: $numLocalNets, numGlobalNets: $numGlobalNets"
    return "$numLocalNets $numGlobalNets"
}


proc getInstTerms_pt_x {instTerms_cur} {
    set instTerms_pt_x ""
    foreach instTerm $instTerms_cur {
        lappend instTerms_pt_x [dbGet $instTerm.pt_x]
    }
    return $instTerms_pt_x
}


proc getInstTerms_pt_y {instTerms_cur} {
    set instTerms_pt_y ""
    foreach instTerm $instTerms_cur {
        lappend instTerms_pt_y [dbGet $instTerm.pt_y]
    }
    return $instTerms_pt_y
}


# ******************************** proc's end **********************************



# ******************************** Analyze placed design ***********************


# Load placed design
source design_placed.enc

# ***************** General, global information on the design ******************
# EDI System Text Command Reference p. 846
reportNetStat
# EDI System Text Command Reference p. 838
report_area -out_file reports/placed/area.txt

# Select Partition x1 y1 x2 y2
# setSelectedPtnCut $box_cur part_001
# Encounter Cmd reference p.887
summaryReport -noHtml -outfile reports/placed/summaryReport.txt
checkDesign -all -nohtml -outfile reports/placed/summaryReport.txt


# ***************** General settings *******************************************
setDbGetMode -displayFormat table -displaylimit 100

# ***************** Loop over all quadrants of dim box_size ********************

# Use of dbu (data base units), as in klayout_scripts.py,
# with 'dbGet -d' gets integer dbu values is unfortunately not possible
# since verify commands expect floats as area coord


# Set variables as in Klayout/klayout_scripts.py to cover the same boxes
set layout_dim_width [dbGet top.fPlan.box_sizex]
set layout_dim_height [dbGet top.fPlan.box_sizey]
# Number of squares on x axis, + 1 to include overlap
# ceil(x): Least integral value greater than or equal to x.
set n_i_float [expr ceil($layout_dim_width / $box_size)]
# Convert to integer
scan $n_i_float %d n_i
# Number of squares on y axis, + 1 to include overlap
set n_j_float [expr ceil($layout_dim_height / $box_size)]
# Convert to integer
scan $n_j_float %d n_j
puts stdout "Querying data from $n_i boxes on x axis and $n_j boxes on y axis."
set x_start [dbGet top.fPlan.box_llx]
set y_start [dbGet top.fPlan.box_lly]


if {$analyze_placed} {
    puts "\n\n**************** Start analyzing placed design *******************"
    # Initialize CSV file with general data about each box
    set csv_string ""
    # CSV heading
    set csv_heading "box_name i j numInsts numInstTerms numLocalNets \
        numGlobalNets numClockTerms"
    set csv_string [csv_append_line $csv_string $csv_heading]

    set x $x_start
    for { set i 0} {$i < $n_i} {incr i} {
        set y $y_start
        for {set j 0} {$j < $n_j} {incr j} {
            # Set variables for each run of analysis
            set left $x
            set bottom $y
            set top [expr $x + $box_size]
            set right [expr $y + $box_size]
            # Use '"' for grouping to ensure substitution of variables
            set box_cur "$left $bottom $top $right"
            set box_name "box_i${i}_j${j}"
            puts "\nCoordinates (um) of current quadrant '$box_name' are: $box_cur"
            set layer_cur {M1 M2 M3 M4 M5}

            # Collect and write data for CSV file with general info about each box
            # Select all instances in current quadrant
            set insts_cur [dbQuery -area $box_cur -objType {inst}]

            # Collect box-specific information
            set numInsts [llength $insts_cur]
            set numInstTerms [get_NumInstTerms $insts_cur]
            # Number of local and global Nets
            set global_local_nets [get_NumLocalGlobalNets $box_cur]
            set numLocalNets  [lindex $global_local_nets 0]
            set numGlobalNets  [lindex $global_local_nets 1]
            # Write box-specific information to CSV file
            set csv_line_list "$box_name $i $j $numInsts $numInstTerms \
                $numLocalNets $numGlobalNets"
            set csv_string [csv_append_line $csv_string $csv_line_list]

            # Get and write detailed data for current quadrant
            area_data_base_query_placed $box_cur $box_name
            set y [expr $y + $box_size]
        }
        set x [expr $x + $box_size]
    }
    # Write CSV file to disk
    puts "\n\nRecorded the following data fields in one overview file: $csv_heading"
    write_string $csv_string reports/placed/box_info.csv
}


# ******************************** Analyze routed design ***********************
if {$analyze_routed} {
    source design_routed.enc

    puts "\n\n**************** Start analyzing routed design **********************"
    # ***************** General, global information on the design ******************
    # EDI System Text Command Reference p. 838
    report_area -out_file reports/routed/area.txt

    # Encounter Cmd reference p.887
    summaryReport -noHtml -outfile reports/routed/summaryReport.txt

    checkDesign -all -nohtml -outfile reports/routed/summaryReport.txt

    # Initialize CSV file with general data about each box
    set csv_string ""
    # CSV heading
    set csv_heading "box_name i j numShorts"
    set csv_string [csv_append_line $csv_string $csv_heading]

    # TODO Separate creation of list with box_cur and box_name into separate proc
    set x $x_start
    for { set i 0} {$i < $n_i} {incr i} {
        set y $y_start
        for {set j 0} {$j < $n_j} {incr j} {
            # Set variables for each run of analysis
            set left $x
            set bottom $y
            set top [expr $x + $box_size]
            set right [expr $y + $box_size]
            # Use '"' for grouping to ensure substitution of variables
            set box_cur "$left $bottom $top $right"
            set box_name "box_i${i}_j${j}"
            puts "\nCoordinates (um) of current quadrant '$box_name' are: $box_cur"
            set layer_cur {M1 M2 M3 M4 M5}

            # Get and write detailed data for current quadrant
            area_data_base_query_routed $box_cur $box_name

            set y [expr $y + $box_size]
        }
        set x [expr $x + $box_size]
    }
    # Write CSV file to disk
    puts "\n\nRecorded the following data fields in one overview file: $csv_heading"
    write_string $csv_string reports/routed/box_info.csv
}