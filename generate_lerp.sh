#!/bin/bash
rm Results/GeneratedImages/*

rm Results/LerpedVideos/lerp.mp4

python generate_lerp_video.py
ffmpeg -r 30 -f image2 -s 512x512 -i Results/GeneratedImages/image%04d.png  -vcodec libx264 -crf 25  -pix_fmt yuv420p Results/LerpedVideos/lerp.mp4