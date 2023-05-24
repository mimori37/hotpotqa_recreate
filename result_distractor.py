
# pip install openpyxl
import os
import glob
import pandas as pd
import datetime
import re

result = 'result[distractor]'
result_text = result + '.txt'
result_excel = result + '.xlsx'
sheet_name = 'data_sheet'
current = 'C:\\Users\\REMOTE\\hotpot\\MyHotpotQA\\'

def train():
    # 学習プロセスを実行
    print('\n[train]')
    print('='*30)
    command = 'python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0'
    print('running command...\n  {}\n'.format(command))
    os.system(command)
    print('[train] process has done.')

def predict():
    # 推測プロセスを実行
    print('\n[predict]')
    print('='*30)
    date = glob.glob('C:\\Users\\REMOTE\\hotpot\\MyHotpotQA\\HOTPOT-*')[-1][-15:]
    command = 'python main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0 --save HOTPOT-{} --prediction_file dev_distractor_pred.json'.format(date)
    print('running command...\n  {}\n'.format(command))
    os.system(command)
    print('[predict] process has done.\n')

def evaluate():
    print('\n[evaluate]')
    print('='*30)

    # 評価スクリプトを[result_text]に出力
    command = 'python hotpot_evaluate_v1.py dev_distractor_pred.json hotpot_dev_distractor_v1.json > ' + result_text
    print('running command...\n  {}\n'.format(command))
    os.system(command)

    # last_eval, last_dev_loss の取得
    date = glob.glob('C:\\Users\\REMOTE\\hotpot\\MyHotpotQA\\HOTPOT-*')
    if len(date) == 0:
        print('セーブファイルがひとつも見つかりませんでした')
    date = date[-1][-15:]
    print('getting last_eval, last_dev_loss...')
    trainlog_text = current + 'HOTPOT-' + date + '\\log.txt'
    with open(trainlog_text) as f:
        train_lines = f.readlines()
    train_lines = [train_line.strip() for train_line in train_lines]
    train_line = [train_line for train_line in train_lines if 'eval' in train_line][-1].strip('|')
    data = train_line.split('|')
    for datos in data:
        datos = datos.split(':')
        index = datos[0].strip(' ')
        value = datos[1]
        if(index == 'eval'):
            value = value.strip(' ')[:2]
            last_eval = value
            print('last_eval: {}'.format(last_eval))
            last_eval = int(last_eval)
        elif(index == 'dev loss'):
            last_dev_loss = value
            print('last_dev_loss: {}'.format(last_dev_loss))
            last_dev_loss = float(last_dev_loss)
        else:
            print('{}: {}'.format(index, value))

    # スコアの取得
    print('getting scores...')
    with open(result_text) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if '{' in line]
    line = lines[0]
    data = line.split(',')
    index, value = [], []
    print('reshaping scores...')
    for datos in data:
        datos = datos.split(':')
        index.append(datos[0].strip('{ \''))
        value.append(float(datos[1].strip('} ')))
    index.append('last_eval')
    value.append(last_eval)
    index.append('last_dev_loss')
    value.append(last_dev_loss)
    index.append('date')
    value.append(date)
    df = pd.DataFrame(value, index=index)
    print(df)
    print('{} data have reshaped.'.format(len(data)))
    print('[evalate] process has done.')
    return df

def output(df):
    # 結果をExcel出力
    print('\n[output]')
    print('='*30)
    if os.path.isfile(result_excel):
        os.remove(result_excel)
    df.to_excel(result_excel, sheet_name=sheet_name)
    command = 'start ' + result_excel
    print('running command...\n  {}'.format(command))
    os.system(command)
    print('[output] process has done.')

def run(tr=True, pr=True, ev=True):
    # カレントディレクトリ変更
    os.chdir(current)
    print('current directory has changed.')

    if tr is True:
        train()
    if pr is True:
        predict()
    if ev is True:
        output(evaluate())

    print('[{}] has done.'.format(__file__))

def main():
    run(tr=False)

if __name__ == '__main__':
    main()
