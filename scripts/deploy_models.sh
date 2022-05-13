SERVER_IP="narval"
SERVER_PATH="~/scratch/sego/"

rsync -avm --exclude '*.pyc' checkpoints ${SERVER_IP}:${SERVER_PATH}