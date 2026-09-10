#!/bin/bash
# Pack a dataset dir into a zstd tar for upload to a GPU node via create_context.
# usage: pack_dataset.sh data/scl_v1 out.tar.zst
set -e
which zstd >/dev/null || (apt-get install -y -qq zstd >/dev/null 2>&1 || pip install --break-system-packages zstandard)
tar -C "$(dirname $1)" -cf - "$(basename $1)" | zstd -T4 -3 -o "$2" -f
sha256sum "$2"; stat -c %s "$2"
