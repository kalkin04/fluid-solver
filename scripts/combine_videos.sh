cd videos
ffmpeg -y -i apic.mp4 -i flip.mp4 -i blend.mp4 -filter_complex hstack=inputs=3 all.mp4