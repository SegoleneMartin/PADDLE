# SERVER_IP="narval"
# SERVER_PATH="~/scratch/sego/"

SERVER_IP="shannon"
SERVER_PATH="/ssd/repos/Few-Shot-Classification/sego"

rsync -avm --exclude '*.pyc' checkpoints/mini ${SERVER_IP}:${SERVER_PATH}/checkpoints/