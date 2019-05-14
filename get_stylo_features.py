import os, sys
import tltk, deepcut
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import re
import string
from collections import Counter
import pickle, datetime


def get_stylo_features(novel, token=True, pos=True, total_sentence = True):

    time_now = datetime.datetime.now().strftime('%Y%m%d-%H:%M')
    print(time_now)
    if token is True:
        print("tokenizing...")
        token = deepcut.tokenize(novel)

    if pos is True:
        print("pos -ing...")
        if total_sentence is True:

            try:
                pos = tltk.nlp.pos_tag(novel)
            except Exception:
                try:
                    pos = tltk.nlp.pos_tag_wordlist(token)
                except Exception:
                    pass

        else:
            pos = tltk.nlp.pos_tag_wordlist(token)

    # N
    f01 = len(token)

    # V
    f02 = len(set(token))

    # Average word len
    sum_word_len = 0
    for i in token:
        sum_word_len += len(i)
    f03 = sum_word_len / len(token)

    # sd of word len
    sum_sq_word_len = 0
    for i in token:
        sum_sq_word_len += (len(i) - f03) ** 2
    f04 = np.sqrt(sum_sq_word_len)

    # V/N
    f05 = f02 / f01

    # VR(K)
    df_word_frequency = pd.DataFrame()
    df_word_frequency["word"] = []
    df_word_frequency["f"] = []
    df_word_frequency["chars"] = []
    df_word_frequency["english_chars"] = []
    df_word_frequency["thai_chars"] = []
    df_word_frequency["thai_num"] = []
    df_word_frequency["digit_num"] = []
    df_word_frequency["special_chars"] = []
    df_word_frequency["thai_vowel"] = []

    total_word_sq = 0
    for i in set(token):
        df_word_frequency.loc[i, 'word'] = i
        df_word_frequency.loc[i, 'f'] = token.count(i)
        df_word_frequency.loc[i, 'chars'] = len(i)
        df_word_frequency.loc[i, "english_chars"] = len(re.findall('[a-zA-Z]', i))
        df_word_frequency.loc[i, "thai_chars"] = len(re.findall('[\u0E00-\u0E4F]', i))
        df_word_frequency.loc[i, "thai_num"] = len(re.findall('[\u0E50-\u0E59]', i))
        df_word_frequency.loc[i, "digit_num"] = len(re.findall('[0-9]', i))
        df_word_frequency.loc[i, "special_chars"] = len(re.findall('[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', i))
        df_word_frequency.loc[i, "thai_vowel"] = len(re.findall('[\u0E30-\u0E4E]', i))

        total_word_sq += np.power(token.count(i), 2)
    f06 = 1e4 * (total_word_sq - f01) / float(np.power(f01, 2))

    # VR(R)
    f07 = f02 / np.sqrt(f01)

    # VR(C)
    f08 = np.log(f02) / np.log(f01)

    # VR(H)
    V1 = len(df_word_frequency[df_word_frequency['f'] == 1])
    f09 = 100 * np.log(f01) / ((1 - V1) / f02)

    # VR(S)
    V2 = len(df_word_frequency[df_word_frequency['f'] == 2])
    f10 = V2 / f02

    # VR(k)
    f11 = np.log(f02) / np.log(np.log(f01))

    # VR(LN)
    f12 = (1 - f02 ** 2) / (f02 ** 2 * np.log(f01))

    # Entropy of word freq distribution
    sum_entropy = 0.
    for i in range(len(df_word_frequency)):
        val = df_word_frequency.iloc[i, 2]
        sum_entropy += (val / f01) * np.log(val / f01)

    f13 = -100 * sum_entropy

    # Total number of chars
    f14 = sum(df_word_frequency['f'] * df_word_frequency['chars'])

    # frequency of alpha(english) chars
    f15 = sum(df_word_frequency['f'] * df_word_frequency['english_chars'])

    # *frequency of thai chars
    f16 = sum(df_word_frequency['f'] * df_word_frequency['thai_chars'])

    # *frequency of thai numeric
    f17 = sum(df_word_frequency['f'] * df_word_frequency['thai_num'])

    # frequency of digitial numeric
    f18 = sum(df_word_frequency['f'] * df_word_frequency['digit_num'])

    # frequency of special chars
    f19 = sum(df_word_frequency['f'] * df_word_frequency['special_chars'])

    # frequency of white spaces
    f20 = len(re.findall('\s', novel))

    # *frequency of thai vowel and tone mark
    f21 = sum(df_word_frequency['f'] * df_word_frequency['thai_vowel'])

    # ratio alpha char (f15)
    f22 = f15 / f14

    # ration thai chars
    f23 = f16 / f14

    # ratio thai num
    f24 = f17 / f14

    # ratio num
    f25 = f18 / f14

    # ratio special chars
    f26 = f19 / f14

    # ratio white spaces
    f27 = f20 / f14

    num_lexical_features = 27

    # POS
    pos_type = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART',
                'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    c = dict()
    if total_sentence is True:
        len_sen = 2
        for x in pos:

            a = Counter([j for i, j in x])
            for b in a:
                if b not in c.keys():
                    c[b] = a[b]
                else:
                    c[b] += a[b]
    else:
        c = Counter([j for i, j in pos])
        len_sen = 0

    # POS as features
    for i in range(len(pos_type)):
        exec('f' + str(num_lexical_features + i + 1) + ' = c[\'' + str(pos_type[i])
             + '\']/len(token) if pos_type[i] in c.keys() else 0')

    i = num_lexical_features + len(pos_type)
    # total_num_sentence
    exec('f' + str(i + 1) + ' = len(pos)')

    # avg_word_per_sentence
    exec('f' + str(i + 2) + ' = len(pos)/f01')

    # Export stylometry list
    stylo_list = []

    for i in range(1, num_lexical_features + len(pos_type) + len_sen + 1):
        # print(i)
        exec('stylo_list.append(f%d%d)' % (np.floor(i/10) % 10, i % 10))

    return stylo_list


def novelize_token(unique_author_name, file_name):
    print('Novelizing...')

    chap_token = []
    novel = ""
    file_novel_name = ''

    for index in range(0, len(file_name)):

        print(file_name.loc[index, 'author_name'], file_name.loc[index, 'novel_name'])

        if file_name.loc[index, 'author_name'] + file_name.loc[index, 'novel_name'] != file_novel_name or \
                index == len(file_name)-1:
            print('saving...')

            if len(chap_token) > 0:
                # save

                # doc
                F = open("..\\all_document\\" + file_novel_name + '.txt', "wt", encoding="utf-8")
                F.writelines(novel)
                F.close()

                # token
                with open("..\\all_token\\" + file_novel_name + '.txt', "wb") as fp:  # Pickling
                    pickle.dump(chap_token, fp, protocol=pickle.HIGHEST_PROTOCOL)

                chap_token = []
                novel = ""

        if file_name.loc[index, 'author_name'] in unique_author_name:
            file_novel_name = file_name.loc[index, 'author_name'] + file_name.loc[index, 'novel_name']

            filename = file_name.loc[index, 'author_name'] + "_" + file_name.loc[index, 'novel_name'] + "_" + \
                       file_name.loc[index, 'chapter_name']

            chap_token += pickle.load(open("..\\token\\" + filename + '.txt', "rb"))
            F = open("..\\document\\" + filename + '.txt', "r", encoding="utf-8")
            novel += F.read()


def run_criteria():
    print('Criteria...')
    cwd = os.getcwd()
    os.chdir('data_raheem_ben')
    os.chdir('document')

    all_file = os.listdir()

    file_name = df()

    file_name['author_name'] = []
    file_name['novel_name'] = []
    file_name['chapter_name'] = []

    for index, file in enumerate(all_file):
        match_1 = re.match('^.*_', file)
        match_2 = re.match('^.*_', match_1[0][:-1])
        novel_name = re.search('_[0-9]*_', match_1[0])

        file_name.loc[index, 'author_name'] = match_2[0][:-1]
        file_name.loc[index, 'novel_name'] = novel_name[0][1:-1]
        file_name.loc[index, 'chapter_name'] = file[-7:-4]

    novel_name = file_name.loc[file_name['novel_name'].drop_duplicates().index, :]
    author_name = novel_name.loc[novel_name['author_name'].duplicated(), :]
    unique_author_name = author_name['author_name'].unique().tolist()

    return unique_author_name, file_name


def chunk_the_novel(window_len = 1000, folder = 'D:\\Thai Stylometry\\data_raheem_ben\\all_token\\', x = 0):

    print('4. Chunking...')
    os.chdir(folder)
    novel_list = os.listdir()
    if x > 0:
        report = df.from_csv('..\\StylometricFeatures.csv')
        run_num = len(report)
    else:
        report = df()
        run_num = 0

    for ii in range(x, len(novel_list)):

        novel_i = novel_list[ii]
        print('ORDER ii: ', ii, novel_i)

        # pickle open
        all_token = pickle.load(open(novel_i, "rb"))
        num_window = np.floor(len(all_token)/window_len)

        for i in range(int(num_window)):

            print(str(i) + '.) ', end="", flush=True)
            token = all_token[window_len * i : min(window_len*(i+1), len(all_token))]
            novel_ = ''.join(token)
            stylo_list = get_stylo_features(novel_, token, pos=True, total_sentence=True)

            # add to df
            file = novel_i

            report.loc[run_num, 'Author'] = file[0:-11]
            report.loc[run_num, 'Novel'] = file[-11:-4]
            report.loc[run_num, 'Chunck'] = i

            for j in range(len(stylo_list)):
                report.loc[run_num, j+1] = stylo_list[j]

            run_num += 1

        # export df to pdf
        report.to_csv('..\\StylometricFeatures.csv')

    return "hello"


if __name__ == '__main__':
    # file = 'C:\\Users\\First ThinkPad\\Desktop\\Thai Stylometry\\data_dek-d_I\\mujakinanao\\1905805\\015.txt'
    # f = open(file, "r", encoding="utf-8")
    # novel = f.read()

    # unique_author_name, file_name = run_criteria()
    # novelize_token(unique_author_name, file_name)
    chunk_the_novel(1000, 'D:\\Thai Stylometry\\data_raheem_ben\\all_token\\', 61)

    #get_stylo_features(novel)

