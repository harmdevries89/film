import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import vr.utils as utils

from logging.handlers import RotatingFileHandler
from torch.autograd import Variable
from vr.data import ClevrDataLoader

def create_logger(save_path):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger

def eval_epoch(loader, model, opt=None):
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    num_samples, num_correct, total_loss = 0.0, 0.0, 0.0
    for batch in loader:
        questions, _, feats, answers, programs, _ = batch
        if isinstance(questions, list):
            questions = questions[0]
        questions_var = Variable(questions.cuda())
        feats_var = Variable(feats.cuda())
        answers_var = Variable(answers.cuda())

        scores = model.forward(questions_var, feats_var)

        loss = loss_fn(scores, answers_var)
        total_loss += loss.cpu().data.numpy()

        _, preds = scores.data.cpu().max(1)
        num_correct += float(sum([pred == tgt for pred, tgt in zip(preds, answers)]))
        num_samples += preds.size(0)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return total_loss/num_samples, num_correct/num_samples


class SimpleFusionModel(nn.Module):

    def __init__(self, wordvec_dim=200, hidden_dim=1024, encoder_vocab_size=30, proj_dim=512, fused_dim=512, num_answers=20):
        super(SimpleFusionModel, self).__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.GRU(wordvec_dim, hidden_dim, batch_first=True)
        self.NULL = 0
        self.num_dir = 1
        self.proj_conv = nn.Conv2d(1024, proj_dim, 1)
        self.proj_layer = nn.Linear(hidden_dim, proj_dim)
        self.film_conv = nn.Conv2d(2*proj_dim, proj_dim, 1)
        self.pool = nn.MaxPool2d(kernel_size=14, stride=14, padding=0)
        self.fused_layer = nn.Linear(proj_dim, fused_dim)
        self.classification_layer = nn.Linear(fused_dim, num_answers)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(proj_dim, affine=True)
        self.bn2 = nn.BatchNorm2d(proj_dim, affine=True)
        self.bn3 = nn.BatchNorm1d(fused_dim, affine=True)

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        mask = Variable(torch.FloatTensor(N, T).fill_(1.0))

        # Find the last non-null element in each sequence.
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
                    mask[i, t + 1:] = 0.0
                    break

        if x.is_cuda:
            mask = mask.cuda()

        x[x.data == self.NULL] = replace
        return x, mask

    def encoder(self, x):
        out = dict()
        x, out['mask'] = self.before_rnn(x)  # Tokenized word sequences (questions), end index
        out['embs'] = self.encoder_embed(x)
        L, N, H = self.encoder_rnn.num_layers * self.num_dir, x.size(0), self.encoder_rnn.hidden_size
        h0 = Variable(torch.zeros(L, N, H).type_as(out['embs'].data))
        out['hs'], _ = self.encoder_rnn(out['embs'], h0)

        return out

    def get_last_state(self, states, mask):
        # Pull out the hidden state for the last non-null value in each input
        N = states.size(0)
        seq_lens = mask.sum(1).long() - 1
        last_hidden_state = states[torch.arange(N).long().cuda(), seq_lens, :]
        return last_hidden_state

    def forward(self, question, image):
        encoded = self.encoder(question)
        last_state = self.get_last_state(encoded['hs'], encoded['mask'])
        projected_question = self.proj_layer(last_state)
        projected_question = projected_question.view(question.size(0), projected_question.size(1), 1, 1).repeat(1, 1, 14, 14)

        projected_image = self.bn1(self.proj_conv(image))

        concat_features = torch.cat([projected_image, projected_question], 1)
        fused_features = self.bn2(self.film_conv(concat_features))
        pooled_features = self.pool(fused_features).squeeze()
        out = self.classification_layer(self.relu(self.bn3(self.fused_layer(pooled_features))))
        return out




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-dir', type=str, default='./exp')
    parser.add_argument('--exp-name', type=str, default='test')

    parser.add_argument('--loader_num_workers', type=int, default=1)

    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=None, type=int)

    # RNN options
    parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
    parser.add_argument('--rnn_hidden_dim', default=512, type=int)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--fused_dim', default=512, type=int)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--reward_decay', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    vocab = utils.load_vocab(os.path.join(args.data_dir, 'vocab.json'))

    model = SimpleFusionModel(encoder_vocab_size=len(vocab['question_token_to_idx']),
                              hidden_dim=args.rnn_hidden_dim,
                              wordvec_dim=args.rnn_wordvec_dim,
                              proj_dim=args.proj_dim,
                              fused_dim=args.fused_dim,
                              num_answers=len(vocab['answer_idx_to_token'])).cuda()

    params = model.parameters()
    opt = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader_kwargs = {
        'question_h5': os.path.join(args.data_dir, 'train_questions.h5'),
        'feature_h5': os.path.join(args.data_dir, 'train_features.h5'),
        'vocab': vocab,
        'batch_size': args.batch_size,
        'shuffle': False,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
        'question_h5': os.path.join(args.data_dir, 'val_questions.h5'),
        'feature_h5': os.path.join(args.data_dir, 'val_features.h5'),
        'vocab': vocab,
        'batch_size': args.batch_size,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers,
    }

    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
            ClevrDataLoader(**val_loader_kwargs) as val_loader:
        best_val_acc = 0.0

        for i in range(30):
            logger.info('Epoch ' + str(i))
            train_loss, train_acc = eval_epoch(train_loader, model, opt=opt)
            valid_loss, valid_acc = eval_epoch(val_loader, model)

            if train_loss.ndim == 1:
                train_loss = train_loss[0]
                valid_loss = valid_loss[0]
            logger.info("{}, {}, {}, {}".format(train_loss, train_acc, valid_loss, valid_acc))

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                state = dict()
                state['parameters'] = model.state_dict()
                state['args'] = args
                torch.save(state, os.path.join(exp_dir, 'model.pt'))









