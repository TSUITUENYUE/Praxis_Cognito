taskset -c 0-7,9-$(($(nproc)-1))
python main.py --mode generate --config ./conf/go2.yaml