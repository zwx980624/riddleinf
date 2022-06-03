export CUDA_VISIBLE_DEVICES=0
cd src
python train_bert.py \
	--output_dir ../train_riddle_bert2 \
	--only_test \
	--model_reload_path /home/zwx/data/learning/nlp/riddle/train_riddle_bert2/epoch_1.ckpt \

