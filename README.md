### Usage guide

Setup:

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib
```

Run any of the following to create a fluid simulation video:

```
python3 methods/pic.py
python3 methods/apic.py
python3 methods/flip.py
python3 methods/blend.py
```

Blend is a blend between FLIP and PIC

To combine the videos and have them play side by side, run `bash scripts/combine_videos.sh`
- Note: I don't like how PIC looks so it didnt make the cut here

### TODO

- build the divergence matrix straight from the csr constructor
- 3D?