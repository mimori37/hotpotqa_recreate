import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model #, NoCharModel, NoSelfModel
from sp_model import SPModel
# from normal_model import NormalModel, NoSelfModel, NoCharModel, NoSentModel
# from oracle_model import OracleModel, OracleModelV2
# from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
nll_sum = nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)    # word_embedding の再利用
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)    # char_embedding の再利用
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)                           # dev_evaluate_fileの再利用
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)                           # index_to_word_dictionary の再利用

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)                     # 同じseed で乱数生成

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))   # time でセーブファイル名を指定
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py', 'sp_model.py'])   # 別の directory にファイルを保存
    def logging(s, print_=True, log_=True):     # print_ オプション, log_ オプション 付きlog関数
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))               # config を全てログ出力

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)       # train, dev 用バケツを用意

    def build_train_iterator():     # イテレータの用意
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, True, config.sent_limit)

    def build_dev_iterator():       # イテレータの用意
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:        # sparse 正則化適用モデルの用意
        model = SPModel(config, word_mat, char_mat)
    else:                           # 通常モデルの用意
        model = Model(config, word_mat, char_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))    # parameter 数
    ori_model = model.cuda()    # original なモデルを保存
    model = nn.DataParallel(ori_model)  # nn 用にモデルを変換

    lr = config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()   # train モード（計算を有効化）

    for epoch in range(10000):
        for data in build_train_iterator():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            q_type = Variable(data['q_type'])
            is_support = Variable(data['is_support'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])         # イテレータのデータ毎に変数を用意

            # logit1: 文章の頭, logit2: 文末, predict_type: 予測タイプ, predict_support: 根拠文 ... を予測
            logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
            loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)     # (損失1) = 予測と正解の損失和 / 2
            loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))                                          # (損失2) = 根拠予測の平均損失
            loss = loss_1 + config.sp_lambda * loss_2   # (総損失) = (損失1) + sp_lambda*(損失2)

            optimizer.zero_grad()   # 勾配の計算を 0 で初期化
            loss.backward()         # loss の計算に用いる変数 (requires_grad=True) の勾配が計算される
            optimizer.step()        # 変数 (requires_grad=True) のパラメータを適当な学習率で更新する

            total_loss += loss.data.item()  # 損失の合計
            global_step += 1                # 次のstepへ

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
                model.train()

                logging('-' * 89)
                logging('| eval: {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss: {:8.3f} | EM: {:.4f} | F1: {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1: # best_dev_F1 が更新されれば
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:  #   設定したlrを下回れば学習終了
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))

def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        with torch.no_grad():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            q_type = Variable(data['q_type'])
            is_support = Variable(data['is_support'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])

        logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        total_loss += loss.data.item()
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss

    return metrics

def predict(data_source, model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    sp_th = config.sp_threshold
    for step, data in enumerate(tqdm(data_source)):

        with torch.no_grad():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            context_lens = Variable(data['context_lens'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])

        logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})

    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)

def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
        print('dev_eval_fileってなんぞや: ---------------------------\n{}'.format(dev_eval_file))
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, para_limit,
            ques_limit, config.char_limit, False, config.sent_limit)

    if config.sp_lambda > 0:
        model = SPModel(config, word_mat, char_mat)
    else:
        model = Model(config, word_mat, char_mat)
    ori_model = model.cuda()
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    model = nn.DataParallel(ori_model)

    model.eval()
    predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)

