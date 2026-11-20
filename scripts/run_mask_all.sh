#!/bin/bash
cd /tmp
N=32458; K=4; per=$(( (N + K - 1) / K ))
for k in $(seq 0 $((K-1))); do
  f0=$((k*per)); n=$per; [ $((f0+n)) -gt $N ] && n=$((N-f0))
  python3 /home/user/work/scripts/mask_worker.py /home/user/work/src/lecture.mp4 $f0 $n /home/user/work/out/masked_$k.mp4 /home/user/work/out/masked_$k.log > /home/user/work/out/worker_$k.txt 2>&1 &
done
wait
cd /home/user/work/out && printf "file 'masked_%d.mp4'\n" 0 1 2 3 > list.txt && ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c copy masked_full.mp4
ffprobe -v error -select_streams v -show_entries stream=nb_frames -of csv=p=0 masked_full.mp4
echo ALLDONE
