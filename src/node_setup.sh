#!/bin/bash
# Bootstrap a givemeanode pytorch node: deps + dataset.
set -e
pip install -q timm pillow numpy scipy requests 2>&1 | tail -1
python -c "import torch, timm; print(torch.__version__, timm.__version__, torch.cuda.is_available())"
if [ -n "$DATA_URL" ] && [ ! -d ~/data/scl_v1 ]; then
  mkdir -p ~/data && cd ~/data && curl -sS -o scl_v1.tar.zst "$DATA_URL" && tar --use-compress-program=unzstd -xf scl_v1.tar.zst && rm scl_v1.tar.zst && ls
fi
