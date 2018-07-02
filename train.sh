export PYTHONPATH=./:/home/pengshanzhen/caffe/python:$PYTHONPATH
LOG=/data_2/my_bishe_experiment/switch_classification/snapshots_64/log-`date +%Y-%m-%d-%H-%M-%S`.log 
/home/pengshanzhen/caffe/build/tools/caffe train -solver solver.prototxt -weights /data_2/my_bishe_experiment/switch_classification/model_iter_10000.caffemodel  2>&1  | tee $LOG $@






