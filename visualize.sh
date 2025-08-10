taskset -c 0-7,9-23,25-$(($(nproc)-1))
python visualizer.py --config ./conf/go2.yaml