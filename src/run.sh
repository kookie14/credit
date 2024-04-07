dataset=$1
cd /home/cuctt/credit/src
export HYDRA_FULL_ERROR=1
python train.py data/dataset=$1