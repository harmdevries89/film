import argparse
import vr.utils as utils
import torch
import torch.optim as optim

from torch.autograd import Variable
from vr.models import FiLMedNet
from vr.models import FiLMGen

from vr.data import ClevrDataLoader

def eval_epoch(loader, film_gen, filmed_net, opt=None):
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    num_samples, num_correct, total_loss = 0.0, 0.0, 0.0
    for batch in loader:
        questions, _, feats, answers, programs, _ = batch
        if isinstance(questions, list):
            questions = questions[0]
        questions_var = Variable(questions.cuda())
        feats_var = Variable(feats.cuda())
        answers_var = Variable(answers.cuda())


        film_params = film_gen.forward(questions_var)
        scores = filmed_net.forward(feats_var, film_params)

        loss = loss_fn(scores, answers_var)
        total_loss += loss.cpu().data.numpy()

        _, preds = scores.data.cpu().max(1)
        num_correct += sum([pred == tgt for pred, tgt in zip(preds, answers)])
        num_samples += preds.size(0)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return total_loss/num_samples, num_correct/num_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--train_question_h5', default='data/train_questions.h5')
    parser.add_argument('--train_features_h5', default='data/train_features.h5')
    parser.add_argument('--val_question_h5', default='data/val_questions.h5')
    parser.add_argument('--val_features_h5', default='data/val_features.h5')
    parser.add_argument('--feature_dim', default='1024,14,14')
    parser.add_argument('--vocab_json', default='data/vocab.json')

    parser.add_argument('--loader_num_workers', type=int, default=1)

    parser.add_argument('--family_split_file', default=None)
    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=None, type=int)
    parser.add_argument('--shuffle_train_data', default=1, type=int)

    # Module net / FiLMedNet options
    parser.add_argument('--module_stem_num_layers', default=2, type=int)
    parser.add_argument('--module_stem_batchnorm', default=0, type=int)
    parser.add_argument('--module_dim', default=128, type=int)
    parser.add_argument('--module_residual', default=1, type=int)
    parser.add_argument('--module_batchnorm', default=0, type=int)

    # FiLM only options
    parser.add_argument('--set_execution_engine_eval', default=0, type=int)
    parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
    parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
    parser.add_argument('--bidirectional', default=0, type=int)
    parser.add_argument('--encoder_type', default='gru', type=str,
      choices=['linear', 'gru', 'lstm'])
    parser.add_argument('--decoder_type', default='linear', type=str,
      choices=['linear', 'gru', 'lstm'])
    parser.add_argument('--gamma_option', default='linear',
      choices=['linear', 'sigmoid', 'tanh', 'exp'])
    parser.add_argument('--gamma_baseline', default=1, type=float)
    parser.add_argument('--num_modules', default=4, type=int)
    parser.add_argument('--module_stem_kernel_size', default=3, type=int)
    parser.add_argument('--module_stem_stride', default=1, type=int)
    parser.add_argument('--module_stem_padding', default=None, type=int)
    parser.add_argument('--module_num_layers', default=1, type=int)  # Only mnl=1 currently implemented
    parser.add_argument('--module_batchnorm_affine', default=0, type=int)  # 1 overrides other factors
    parser.add_argument('--module_dropout', default=5e-2, type=float)
    parser.add_argument('--module_input_proj', default=1, type=int)  # Inp conv kernel size (0 for None)
    parser.add_argument('--module_kernel_size', default=3, type=int)
    parser.add_argument('--condition_method', default='bn-film', type=str,
      choices=['block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
    parser.add_argument('--condition_pattern', default='', type=str)  # List of 0/1's (len = # FiLMs)
    parser.add_argument('--use_gamma', default=1, type=int)
    parser.add_argument('--use_beta', default=1, type=int)
    parser.add_argument('--use_coords', default=1, type=int)  # 0: none, 1: low usage, 2: high usage
    parser.add_argument('--grad_clip', default=0, type=float)  # <= 0 for no grad clipping
    parser.add_argument('--debug_every', default=float('inf'), type=float)  # inf for no pdb
    parser.add_argument('--print_verbose_every', default=float('inf'), type=float)  # inf for min print

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_iterations', default=100000, type=int)
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--reward_decay', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    # Output options
    parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
    parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
    parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
    parser.add_argument('--record_loss_every', default=1, type=int)
    parser.add_argument('--checkpoint_every', default=10000, type=int)
    parser.add_argument('--time', default=0, type=int)

    args = parser.parse_args()

    vocab = utils.load_vocab(args.vocab_json)

    train_loader_kwargs = {
        'question_h5': args.train_question_h5,
        'feature_h5': args.train_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'shuffle': False,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
        'question_h5': args.val_question_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers,
    }

    film_gen = FiLMGen().cuda()
    filmed_net = FiLMedNet(vocab).cuda()

    params = list(film_gen.parameters()) + list(filmed_net.parameters())
    opt = optim.Adam(params)

    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
            ClevrDataLoader(**val_loader_kwargs) as val_loader:
        for _ in range(100):
            train_loss, train_acc = eval_epoch(train_loader, film_gen, filmed_net, opt=opt)
            print(train_loss, train_acc)

