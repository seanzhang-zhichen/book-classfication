
import time
import torch
import numpy as np
import pandas as pd
from importlib import import_module
import argparse
from torch.utils.data import DataLoader
import joblib
from dataset import MyDataset, collate_fn
from train_helper import train, init_network
from dictionary import Dictionary
from config import create_logger, root_path, device, train_file, bert_path, dev_file, batch_size, test_file

from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer


parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument(
#     '--model',
#     type=str,
#     required=True,
#     help='choose a model: CNN, RNN, RCNN, RNN_Att, DPCNN, Transformer')
parser.add_argument('--word',
                    default=True,
                    type=bool,
                    help='True for word, False for char')
parser.add_argument('--max_length',
                    default=400,
                    type=int,
                    help='True for word, False for char')
parser.add_argument('--dictionary',
                    default=None,
                    type=str,
                    help='dictionary path')
args = parser.parse_args()

logger = create_logger(root_path + '/logs/main.log')

if __name__ == '__main__':
    model_name = 'bert'

    x = import_module('models.' + model_name)
    # if model_name in ['bert', 'xlnet', 'roberta']:
    #     config.bert_path = config.root_path + '/model/' + model_name + '/'
    #     if 'bert' in model_name:
    #         config.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    #     elif 'xlnet' in model_name:
    #         config.tokenizer = XLNetTokenizer.from_pretrained(config.bert_path)
    #     elif 'roberta' in model_name:
    #         config.tokenizer = RobertaTokenizer.from_pretrained(config.bert_path)
    #     else:
    #         raise NotImplementedError
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    save_path = root_path + '/model/' + model_name + '.ckpt'  # 模型训练结果
    log_path = root_path + '/logs/' + model_name
    hidden_size = 768
    eps = 1e-8
    gradient_accumulation_steps = 1
    word = True
    max_length = 400
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logger.info("Loading data...")

    logger.info('Building dictionary ...')

    data = pd.read_csv(train_file, sep='\t')
    if args.word:
        data = data['text'].values.tolist()
    else:
        data = data['text'].apply(lambda x: " ".join("".join(x.split())))
    if args.dictionary is None:
        dictionary = Dictionary()
        dictionary.build_dictionary(data)
        del data
        joblib.dump(dictionary, root_path + '/model/vocab.bin')
    else:
        dictionary = joblib.load(args.dictionary)

    tokenizer = tokenizer


    logger.info('Making dataset & dataloader...')
    train_dataset = MyDataset(train_file,
                              dictionary,
                              args.max_length,
                              tokenizer=tokenizer,
                              word=args.word)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    dev_dataset = MyDataset(dev_file,
                            dictionary,
                            args.max_length,
                            tokenizer=tokenizer,
                            word=args.word)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collate_fn)
    test_dataset = MyDataset(test_file,
                             dictionary,
                             args.max_length,
                             tokenizer=tokenizer,
                             word=args.word)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn)

    # train

    #     conf.n_vocab = dictionary.max_vocab_size
    model = x.Model().to(device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    train(model, train_dataloader, dev_dataloader, test_dataloader)
