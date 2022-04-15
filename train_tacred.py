import argparse
import os

import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig, AlbertTokenizer, AlbertConfig, AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import TACREDProcessor
from evaluation import get_f1
from model.model import REModel
from torch.cuda.amp import GradScaler
import wandb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

dev_result_list = []
test_result_list = []
test_pred_list = []

rev_dev_result_list = []
rev_test_result_list = []
rev_test_pred_list = []

def train(args, processor, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    eval_step = args.evaluation_steps
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      's_mask': batch[3].to(args.device),
                      'o_mask': batch[4].to(args.device),
                      }
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                # wandb.log({'loss': loss.item()}, step=num_steps)

            if (num_steps % eval_step == 0 and (step + 1) % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    pred, f1, result = evaluate(epoch, args, model, features, tag=tag)
                    # wandb.log(output, step=num_steps)
                    logger.info(result)
                    if tag == 'dev':
                        dev_result_list.append(result[tag + "_f1"])
                    elif tag == 'test':
                        test_result_list.append(result)
                        test_pred_list.append(pred)
                    elif tag == 'dev_rev':
                        rev_dev_result_list.append(result[tag + "_f1"])
                    elif tag == 'test_rev':
                        rev_test_result_list.append(result)
                        rev_test_pred_list.append(pred)

    for tag, features in benchmarks:
        pred, f1, result = evaluate(args.num_train_epochs, args, model, features, tag=tag)
        # wandb.log(output, step=num_steps)
        logger.info(result)
        if tag == 'dev':
            dev_result_list.append(result[tag + "_f1"])
        elif tag == 'test':
            test_result_list.append(result)
            test_pred_list.append(pred)
        elif tag == 'dev_rev':
            rev_dev_result_list.append(result[tag + "_f1"])
        elif tag == 'test_rev':
            rev_test_result_list.append(result)
            rev_test_pred_list.append(pred)



def evaluate(num_epochs, args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  's_mask': batch[3].to(args.device),
                  'o_mask': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    max_prec, max_recall, max_f1 = get_f1(keys, preds)

    output = {
        "epochs": num_epochs,
        tag + "_prec": max_prec * 100,
        tag + "_recall": max_recall * 100,
        tag + "_f1": max_f1 * 100,
    }
    print(output)

    return preds, max_f1, output



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--output_dir", default="./saved_models/tacred_roberta", type=str,
                        help="model save.")
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--evaluation_steps", type=int, default=500,
                         help="Number of steps to evaluate the model")

    parser.add_argument("--eval_metric", default="f1", type=str)


    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="MD-AGDPN")
    parser.add_argument("--run_name", type=str, default="tacred")

    parser.add_argument("--graph_hidden_size", default=300, type=int,
                        help="graph_hidden_size")
    parser.add_argument('--input_dropout',
                        type=float, default=0.5,
                        help='Dropout rate for word representation.')
    parser.add_argument('--graph_layers',
                        type=int,
                        default=2,
                        help="graph_layers")

    parser.add_argument('--heads',
                        type=int,
                        default=3,
                        help="heads")

    parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers.')
    parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers.')

    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN dropout.')

    parser.add_argument('--gcn_model', type=str, default='multi_gcn', help='gcn model name')

    parser.add_argument('--pooling_ratio', type=float, default=0.7,
                        help='pooling ratio')

    args = parser.parse_args()
    # wandb.init(project=args.project_name, name=args.run_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)


    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    processor = TACREDProcessor(args, tokenizer)

    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=processor.num_labels,
    )
    config.gradient_checkpointing = True

    model = REModel(args, config, processor.num_labels)
    model.to(0)

    train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")
    dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
    test_rev_file = os.path.join(args.data_dir, "test_rev.json")

    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    dev_rev_features = processor.read(dev_rev_file)
    test_rev_features = processor.read(test_rev_file)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        ("dev_rev", dev_rev_features),
        ("test_rev", test_rev_features),
    )
    train(args, processor, model, train_features, benchmarks)

    index = dev_result_list.index(max(dev_result_list))
    pred = test_pred_list[index]
    result = test_result_list[index]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info(result)
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for ex, pred in zip(test_features, pred):
            f.write("%s\t%s\n" % (ex['guid'], processor.id2label[pred]))
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key in sorted(result.keys()):
            f.write("%s = %s\n" % (key, str(result[key])))

    rev_index = rev_dev_result_list.index(max(rev_dev_result_list))
    rev_pred = rev_test_pred_list[rev_index]
    rev_result = rev_test_result_list[rev_index]

    logger.info(rev_result)
    with open(os.path.join(args.output_dir, "predictions_rev.txt"), "w") as f_rev:
        for rev_ex, rev_pred in zip(test_rev_features, rev_pred):
            f_rev.write("%s\t%s\n" % (rev_ex['guid'], processor.id2label[rev_pred]))
    with open(os.path.join(args.output_dir, "test_results_rev.txt"), "w") as f_rev:
        for rev_key in sorted(rev_result.keys()):
            f_rev.write("%s = %s\n" % (rev_key, str(rev_result[rev_key])))



if __name__ == "__main__":
    main()
