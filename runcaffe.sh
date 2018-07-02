export PYTHONPATH=./:/home/pengshanzhen/caffe/python:$PYTHONPATH
LOG=/data_2/my_bishe_experiment/switch_classification/three_class/snapshots/log-`date +%Y-%m-%d-%H-%M-%S`.log 
/home/pengshanzhen/caffe/build/tools/caffe train -solver solver.prototxt  2>&1  | tee $LOG $@






