export CUDA_VISIBLE_DEVICES=0
cd src
python train_bert.py \
        --output_dir ../train_bert_base \
	--train_file ../data/train.csv \
	--dev_file ../data/valid_small.csv \
	--test_file ../data/test.txt \
	--n_val 5 \
	--n_epochs 30
