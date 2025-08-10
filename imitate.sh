taskset -c 0-1,3-7,10-$(($(nproc)-1)) python main.py --mode imitate --config ./conf/go2.yaml --demo "Pretrain/data/go2/demo/walk.h5"
