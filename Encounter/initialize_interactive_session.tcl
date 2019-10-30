# Use the enc_tcl_return_display_limit variable to control the length before truncation
set enc_tcl_return_display_limit 500
setDbGetMode -displayFormat table -displaylimit 100
# CHECK Load correct design (placed or routed)
source design_routed.enc
set box_cur {60 60 66 66}
set markers_cur [dbQuery -area $box_cur -objType {marker}]
set nets_cur [dbQuery -area $box_cur -objType {net}]
set insts_cur [dbQuery -area $box_cur -objType {inst}]
set inst_cur [lindex $insts_cur 1]