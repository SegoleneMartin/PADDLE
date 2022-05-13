SERVER_IP="narval"
SERVER_PATH="~/scratch/sego/"

rsync -avm --exclude '*.pyc' src ${SERVER_IP}:${SERVER_PATH}
rsync -avm --exclude '*.pyc' scripts ${SERVER_IP}:${SERVER_PATH} 
rsync -avm --exclude '*.pyc' config ${SERVER_IP}:${SERVER_PATH}