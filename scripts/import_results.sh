SERVER_IP="shannon"
SERVER_PATH="/ssd/repos/Few-Shot-Classification/sego"

rsync -av ${SERVER_IP}:${SERVER_PATH}/results_test/ ./results_test/

rsync -av ${SERVER_IP}:${SERVER_PATH}/results_ablation/ ./results_ablation/