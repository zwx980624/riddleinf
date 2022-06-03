export CUDA_VISIBLE_DEVICES=0
cd src
python train_bert.py \
	--output_dir ../train_riddle_bert_recall \
	--use_recall \
