#!/bin/bash

SCRIPT1="AMAZON"
SCRIPT2="MAG"

echo "=== Auto-Shutdown Monitor Started ==="
echo "Monitoring processes: $SCRIPT1 and $SCRIPT2"
echo "---------------------------------------"
echo ""

while true; do
    RUN1=$(pgrep -f "$SCRIPT1")
    RUN2=$(pgrep -f "$SCRIPT2")

    if [[ -n "$RUN1" && -n "$RUN2" ]]; then
        echo "[INFO] Both processes are running."
        echo "--- $SCRIPT1 details ---"
        ps -fp "$RUN1"
        echo "--- $SCRIPT2 details ---"
        ps -fp "$RUN2"
        echo ""
    elif [[ -n "$RUN1" ]]; then
        echo "[INFO] $SCRIPT1 is running. $SCRIPT2 has finished."
        ps -fp "$RUN1"
        echo ""
    elif [[ -n "$RUN2" ]]; then
        echo "[INFO] $SCRIPT2 is running. $SCRIPT1 has finished."
        ps -fp "$RUN2"
        echo ""
    else
        echo "[DONE] Both processes have finished."
        echo "Shutting down the instance..."
        sudo shutdown -h now
        exit
    fi

    sleep 60
done

