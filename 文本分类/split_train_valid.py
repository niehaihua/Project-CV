import random

'''切割训练集和验证集'''
'''切割训练集和验证集'''


def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    # 计算lines列表的长度，即行数
    num_lines = len(lines)
    # 计算训练集的行数
    num_train = int(0.8 * num_lines)

    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    with open('train.txt', 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)

    with open('valid.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)


split_file(r'文本分类练习.csv')