#!/bin/bash
while pgrep -f "bl_render_clips.py -- clavicle" >/dev/null; do sleep 10; done
cd /tmp && RES_W=720 RES_H=810 SPP=12 nice -n 5 blender -b --python /home/user/work/scripts/bl_render_clips.py -- clavicle cage scapula humerus forearm hand position directions views features 2>&1 | grep -E "Error|Traceback|ALLCLIPS|FRAME .* 0 "
