# SERVER_IP="narval"
# SERVER_PATH="~/scratch/sego/"


SERVER_IP="shannon"
SERVER_PATH="/ssd/repos/Few-Shot-Classification/sego"


rsync -av  \
  --exclude .git \
  --exclude logs \
  --exclude data \
  --exclude results_test \
  --exclude results_test_old \
  --exclude results_ablation \
  --exclude archive \
  --exclude checkpoints \
  --exclude *.tar \
  --exclude training.log \
  --exclude results \
  --exclude __pycache__ \
  --exclude tmp \
  --exclude *.sublime-project \
  --exclude *.sublime-workspace \
  --exclude output \
  --exclude *.md \
  --exclude plots \
  --exclude *.so \
  ./ ${SERVER_IP}:${SERVER_PATH}/

  