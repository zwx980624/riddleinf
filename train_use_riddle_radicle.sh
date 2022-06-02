export CUDA_VISIBLE_DEVICES=0
cd src
python train_bert.py \
	--output_dir ../train_riddle_bert_riddle_radicle \
	--use_riddle_radicle \
