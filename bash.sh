nohup python -u wavlm_train1.py\
  --datadir /home/duhuipeng/SAMOS/data/phase1-main/DATA \
  --wavlm_model /home/duhuipeng/SAMOS/wavlm-base-plus \
  --outdir ckpt/wavlmbase/1 \
  --gpu 2 \
  --batch_size 2 \
  --learning_rate 0.0001 \
  --max_epochs 100 \
  --fine_tune_ssl \
  > wavlm_training.log 2>&1 &