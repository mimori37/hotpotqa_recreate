# pip install openpyxl
import os
import glob
import time # 結果出力のカッコつけのため
import pandas as pd
import datetime
import re

trainlog_text = 'trainlog[distractor].txt'
result = 'result[distractor]'
result_text = result + '.txt'
result_excel = result + '.xlsx'
sheet_name = 'data_sheet'

current = 'C:/Users/REMOTE/hotpot/MyHotpotQA/'

def main():
    # カレントディレクトリ変更
    os.chdir(current)
    print('current directory has changed.')
    time.sleep(0.2)

    date = glob.glob('C:\\Users\\REMOTE\\hotpot\\MyHotpotQA\\HOTPOT-*')[-1][-15:]

    # last_eval, last_dev_loss の取得
    print('get last_eval, last_dev_loss')
    with open(trainlog_text) as f:
        train_lines = f.readlines()
    train_lines = [train_line.strip() for train_line in train_lines]
    train_line = [train_line for train_line in train_lines if 'eval' in train_line][-1]
    data = train_line.strip('|').split('|')
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

    # 行のリスト取得
    with open(result_text) as f:
        lines = f.readlines()
    print('results have listed.')

    # 改行文字除去
    lines = [line.strip() for line in lines]
    print('newline_char has removed.')

    # 条件を満たす行の抽出
    lines = [line for line in lines if '{' in line]
    line = lines[0]
    print('data have selected.')
    time.sleep(0.2)

    # 分割
    data = line.split(',')
    index, value = [], []
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
    time.sleep(0.1)
    df = pd.DataFrame(value, index=index)
    print(df)
    print('{} data have reshaped.'.format(len(data)))
    time.sleep(0.2)

    # 結果をExcel出力
    if os.path.isfile(result_excel):
        os.remove(result_excel)
    df.to_excel(result_excel, sheet_name=sheet_name)
    command = 'start ' + result_excel
    os.system(command)

if __name__ == '__main__':
    main()
