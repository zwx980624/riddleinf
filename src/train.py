import sys
import os
import torch
import logging
import argparse
from transformers import BertTokenizer, BertModel, AdamW, AutoTokenizer, AutoModel

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initial_model(args):
    red_emb_in, red_emb_out, red_num_layers = 128, 128, 2
    ans_emb_in, ans_hidden, ans_emb_out = 256, 1024, 256
    model = RiddleModel(args, red_emb_in, red_emb_out, red_num_layers, ans_emb_in, ans_hidden, ans_emb_out)
    
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='', type=str, required=True, help='Model Saved Path, Output Directory')
    parser.add_argument("--bert_pretrain_name", default='bert-base-chinese')
    parser.add_argument('--bert_pretrain_path', default='', type=str, required=True)
    parser.add_argument('--train_file', default='', type=str, required=True)

    parser.add_argument('--model_reload_path', default='', type=str, help='pretrained model to finetune')
    parser.add_argument('--finetune_from_trainset', default='', type=str, help='train_file which pretrained model used, important for alignment output vocab')

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dev_file', default='Math_23K_mbert_token_val.json', type=str)
    parser.add_argument('--test_file', default='Math_23K_mbert_token_test.json', type=str)

    parser.add_argument('--schedule', default='linear', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--max_grad_norm', default=3.0, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)

    parser.add_argument('--n_save_ckpt', default=1, type=int, help='totally save $n_save_ckpt best ckpts')
    parser.add_argument('--n_val', default=5, type=int, help='conduct validation every $n_val epochs')
    parser.add_argument('--logging_steps', default=100, type=int)

    parser.add_argument('--embedding_size', default=128, type=int, help='Embedding size')
    parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size')
    parser.add_argument('--beam_size', default=5, type=int, help='Beam size')

    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--seed', default=42, type=int, help='universal seed')

    parser.add_argument('--only_test', action='store_true')

    args = parser.parse_args()

    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args)

    if os.path.exists(os.path.join(args.output_dir, "log.txt")) and not args.only_test:
        print("remove log file")
        os.remove(os.path.join(args.output_dir, "log.txt"))
    if args.only_test:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log_test.txt"))
    else:
        handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # train_data, val_data
    train_data
    val_data
    # initialize model
    model = initial_model()

    if torch.cuda.is_available():
        model.cuda()
    if not args.only_test:
        train_model()
    test_model()

if __name__ == "__main__":
    main()