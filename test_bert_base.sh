export CUDA_VISIBLE_DEVICES=0
cd src
python train_bert.py \
	--output_dir ../train_bert_base \
	--only_test \
	--test_file ../data/test.txt \
	--model_reload_path train_bert_base/epoch_1.ckpt \

