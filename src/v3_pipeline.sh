#!/bin/bash
# v3 on the GPU node: masks + tall-structure set + snapped labels + rare sampler. Run from ~ (dev user).
set -e
cd ~/data/scl_v1/src
Y=2025,2023,2021,2019,2017,2015,2013
echo "== 1. snap tall-set labels onto v2 responses (train split), protect the rest with ignore discs"
python3 dump_preds.py --ckpt ~/runs/v2_cnxt/best.pt --data ~/data/tall_set --years $Y --split train --out ~/data/tall_set/preds_v2_train.json 2>&1 | grep -v Warn
python3 snap_labels.py --data ~/data/tall_set --preds ~/data/tall_set/preds_v2_train.json --radius_px 150 --min_score 0.08 --split train
echo "== 2. combine v3"
rm -rf ~/data/v3
python3 combine_sets.py --out ~/data/v3 --val_frac_override 0.3 \
  --set scl:/home/dev/data/scl_v1:index_refined.json \
  --set redmond:/home/dev/data/redmond_eval:index.json:/home/dev/data/redmond_eval/preds_refined.json \
  --set renton:/home/dev/data/renton_set:index.json:/home/dev/data/renton_set/preds_refined.json \
  --set federalway:/home/dev/data/federalway_set:index.json:/home/dev/data/federalway_set/preds_refined.json \
  --set kirkland:/home/dev/data/kirkland_set:index.json:/home/dev/data/kirkland_set/preds_refined.json \
  --set tall:/home/dev/data/tall_set:index.json
ln -sfn ~/data/scl_v1/src ~/data/v3/src
ls ~/data/v3/blocks | grep -c ignore.png
python3 -c "import json; idx=json.load(open('/home/dev/data/v3/index.json')); print('rare train blocks', sum(1 for i in idx if i['split']=='train' and i.get('rare')), 'of', sum(1 for i in idx if i['split']=='train'))"
echo "== 3. train v3 (ConvNeXt-Tiny, ignore masks, rare sampler x4)"
DATA=~/data/v3 ~/run_exp.sh v3_cnxt --years $Y --max_years 4 --year_dropout 0.3 --epochs 60 --bs 12 --lr 2e-4 --backbone convnext_tiny --crop 512 --val_bs 6 --dist_px 20 --eval_every 2 --workers 14 --use_ignore 1 --rare_weight 4
echo "== 4. per-set eval"
for s in scl redmond renton federalway kirkland tall; do echo "== v3_cnxt / $s"; python3 eval_radii.py --ckpt ~/runs/v3_cnxt/best.pt --data ~/data/v3 --years $Y --thresh 0.25 --edge 24 --radii 20,30,50 --only_set $s 2>&1 | grep -v Warn; done
echo "== v2_cnxt / tall (baseline)"; python3 eval_radii.py --ckpt ~/runs/v2_cnxt/best.pt --data ~/data/v3 --years $Y --thresh 0.25 --edge 24 --radii 20,30,50 --only_set tall 2>&1 | grep -v Warn
echo "== 5. probes (fields / parking) with v3"
for spec in "fields:-122.3435,47.6655,-122.3395,47.6685" "parking:-122.3300,47.7040,-122.3260,47.7070"; do n=${spec%%:*}; b=${spec#*:}; python3 infer.py --ckpt ~/runs/v3_cnxt/best.pt --bbox=$b --years 2025,2023,2021 --out ~/viz/probe_v3_$n.geojson --thresh 0.1 2>&1 | grep -v Warn | tail -1; python3 -c "
import json; g=json.load(open('/home/dev/viz/probe_v3_$n.geojson')); s=sorted([f['properties']['score'] for f in g['features']],reverse=True); print('$n', len(s), 'peaks>=0.1;', sum(x>=0.25 for x in s), '>=0.25;', sum(x>=0.5 for x in s), '>=0.5; top', [round(x,2) for x in s[:8]])"; done
echo V3_DONE
