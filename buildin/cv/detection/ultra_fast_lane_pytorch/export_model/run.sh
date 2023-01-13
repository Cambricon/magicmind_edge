#!/bin/bash
set -e
# trace model
if [ -f $PROJ_ROOT_PATH/data/models/tusimple_18_traced.pt ]; then
    echo "tusimple_18_traced.pt model already exists."
else 
    cd $PROJ_ROOT_PATH/export_model/
    echo "export model begin..."
    python export_trace.py tusimple.py --test_model $PROJ_ROOT_PATH/data/models/tusimple_18.pth
    echo "tusimple_18_traced.pt model saved in $PROJ_ROOT_PATH/data/models"
fi
