import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_transformers import BertTokenizer, BertModel
from models.coconut_model import CoconutModel
from project_dataset import ProjectDataset
# from torch.utils.data import DataLoader
from utils.utils import AverageMeter

parser = argparse.ArgumentParser(description='COMP90051 Project1')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--l2_reg', type=float, default=0.00025)
parser.add_argument('--num_of_classes', type=int, default=9292)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v1")
parser.add_argument('--train_set_file_path', type=str, default="data/v1/train_set_v1.txt")
parser.add_argument('--dev_set_file_path', type=str, default="data/v1/dev_set_v1.txt")
parser.add_argument('--idx_file_path', type=str, default="data/v1/v1_idx.pickle")
parser.add_argument('--log_every', type=int, default=1000)

args = parser.parse_args()


def get_data_loader():
    data_loaders = {}

    train_set = ProjectDataset(file_path=args.train_set_file_path,
                               idx_file_path=args.idx_file_path,
                               num_of_classes=args.num_of_classes)

    dev_set = ProjectDataset(file_path=args.dev_set_file_path,
                             idx_file_path=args.idx_file_path,
                             num_of_classes=args.num_of_classes)

    # train_loader = DataLoader(train_set,
    #                           batch_size=1,
    #                           shuffle=False,
    #                           pin_memory=True,
    #                           num_workers=4)
    #
    # dev_loader = DataLoader(dev_set,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         num_workers=4)

    data_loaders["train_set"] = train_set
    data_loaders["dev_set"] = dev_set
    return data_loaders


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_embed(bert_model, indexed_tokens, segments_ids):
    if torch.cuda.is_available():
        bert_model = bert_model.cuda()
        tokens_tensor = torch.tensor(indexed_tokens).cuda()
        segments_tensors = torch.tensor(segments_ids).cuda()
    else:
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
    return outputs[0]


def get_bert_embed(bert_model, tokenizer, sentence):
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(indexed_tokens)

    if len(indexed_tokens) > 512:
        indexed_tokens = list(chunks(indexed_tokens, 512))
        segments_ids = list(chunks(segments_ids, 512))
    else:
        indexed_tokens = [indexed_tokens]
        segments_ids = [segments_ids]

    out_tensor = None
    for i in range(len(indexed_tokens)):
        out = get_embed(bert_model, [indexed_tokens[i]], [segments_ids[i]])
        if out_tensor is None:
            out_tensor = out
        else:
            out_tensor = torch.cat([out_tensor, out], dim=1)
    return out_tensor


def save_mode(epoch, model, optimizer):
    filename = args.check_point_path + args.model_version_string + '.pth'
    payload = {
        "epoch": epoch + 1,
        "args": args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(payload, filename)
    print('\n %s Saved! \n' % (filename))


def load_model(model, optimizer):
    filename = args.check_point_path + args.model_version_string + '.pth'
    checkpoints = torch.load(filename)
    epoch = checkpoints["epoch"]
    model.load_state_dict(checkpoints["model_state_dict"])
    optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    return epoch, model, optimizer


def eval_model(epoch, model, loader, bert_model, tokenizer):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    print('=' * 20 + "Model Eval" + '=' * 20)
    with torch.no_grad():
        pbar = tqdm(loader.samples_frame.iterrows())
        for index, row in pbar:
            start = time.time()
            id, sentence = row.values
            labels = torch.tensor([loader.class_idx[id]])
            embed = get_bert_embed(bert_model, tokenizer, sentence)
            if torch.cuda.is_available():
                labels = labels.cuda(non_blocking=True)
                embed = embed.cuda(non_blocking=True)

            pred = model(embed)
            loss = nn.CrossEntropyLoss()(pred, labels)

            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            acc_meter.update(train_acc.item())
            loss_meter.update(loss.item())
            end = time.time()
            used_time = end - start
            if (index) % args.log_every == 0:
                display = 'epoch=' + str(epoch) + \
                          '\tloss=%.6f' % (loss_meter.val) + \
                          '\tloss_avg=%.6f' % (loss_meter.avg) + \
                          '\tacc=%.4f' % (acc_meter.val) + \
                          '\tacc_avg=%.4f' % (acc_meter.avg) + \
                          '\ttime=%.2fit/s' % (1. / used_time)
                pbar.write(display)

    print("Final Eval Acc %.4f" % acc_meter.avg)
    return


def train_model(epoch, model, optimizer, loader, bert_model, tokenizer):
    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    model.train()
    print('=' * 20 + "Model Training" + '=' * 20)
    pbar = tqdm(loader.samples_frame.iterrows())
    for index, row in pbar:
        start = time.time()
        id, sentence = row.values
        labels = torch.tensor([loader.class_idx[id]])
        embed = get_bert_embed(bert_model, tokenizer, sentence)
        if torch.cuda.is_available():
            labels = labels.cuda(non_blocking=True)
            embed = embed.cuda(non_blocking=True)

        optimizer.zero_grad()
        model.zero_grad()

        pred = model(embed)
        loss = nn.CrossEntropyLoss()(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_bound)
        optimizer.step()

        train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())
        end = time.time()
        used_time = end - start

        if (index) % args.log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tloss_avg=%.6f' % (loss_meter.avg) + \
                      '\tlr=%.6f' % (lr) + \
                      '\tacc=%.4f' % (train_acc_meter.val) + \
                      '\tacc_avg=%.4f' % (train_acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
            pbar.write(display)

    return


def train():
    # Init Training
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    data_loaders = get_data_loader()
    coconut_model = CoconutModel()
    optimizer = torch.optim.Adam(params=coconut_model.parameters(),
                                 lr=args.lr,
                                 betas=(0.0, 0.999),
                                 eps=1e-3,
                                 weight_decay=args.l2_reg)
    starting_epoch = 0

    if args.resume:
        starting_epoch, coconut_model, optimizer = load_model(model=coconut_model,
                                                              optimizer=optimizer)

    if torch.cuda.is_available():
        coconut_model = coconut_model.cuda()

    for epoch in range(starting_epoch, args.epoch):
        train_model(epoch=epoch,
                    model=coconut_model,
                    optimizer=optimizer,
                    loader=data_loaders["train_set"],
                    tokenizer=tokenizer,
                    bert_model=bert_model)
        eval_model(epoch=epoch,
                   model=coconut_model,
                   loader=data_loaders["dev_set"],
                   tokenizer=tokenizer,
                   bert_model=bert_model)
        save_mode(epoch=epoch,
                  model=coconut_model,
                  optimizer=optimizer)
    return


if __name__ == "__main__":
    train()
