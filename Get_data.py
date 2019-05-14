import os
import sys
import pandas as pd
from util import *
from time import sleep
import numpy as np

owd = os.getcwd()

# print(sys.executable)
# print(sys.path)
print(owd)
# df_all = df()
# df_all['Author_name'] = ''
# df_all['Title'] = ''
# df_all['Link'] = ''
#
# for i in range(0, 500):
#     print(i)
#     a = get_novels(i)
#     sleep(0.7)
#     for j in range(len(a)):
#         if len(a[j]) > 40 :
#             df_all.loc[30 *i + j, 'Author_name'] = a[j]['username']
#             df_all.loc[30 *i + j, 'Title'] = a[j]['title']
#             df_all.loc[30 *i + j, 'Link'] = a[j]['id']
#
# df_all.index = range(len(df_all))
# # save df as xlsx
# df_al = df_all.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
# df_al.to_excel('All_novel.xlsx')
#

df_all_data = pd.read_excel('All_novel.xlsx')
# df_all_data = df_all

os.chdir(owd)

# folder level
folder_name = 'data_dek-d_I'
into_folder(folder_name)

for i in range(8862, len(df_all_data)): #5582 #5785new code
    #1586    #5742
    count = 0
    print('>>>>>', i)
    # Author_name
    author_name = df_all_data.loc[i, 'Author_name']
    into_folder(author_name)

    # Title
    title_name = df_all_data.loc[i, 'Link']
    into_folder(title_name)

    print("Author : %6s, Title: %6s " % (author_name, title_name))
    try:
        link_list = get_chapter(df_all_data.loc[i, 'Link'])
    except Exception:
        try:
            print('try1 web error')
            sleep(1)
            link_list = get_chapter(df_all_data.loc[i, 'Link'])
        except Exception:
            try:
                print('try2 web error')
                sleep(1)
                link_list = get_chapter(df_all_data.loc[i, 'Link'])
            except Exception:
                continue

    if link_list is not None or link_list != []:
        print(len(link_list))
        for counter, l in enumerate(link_list, 1):
            if counter%10 == 0 or counter==len(link_list):
                print('X')
            else:
                print('X', end="", flush=True)
            reg_url = "https://writer.dek-d.com/punn3kay/story/" + l
            write_text = None
            sleep(0.05*np.random.rand())
            try:
                write_text = get_text(reg_url)
            except Exception: #TimeoutError, Exception
                try:
                    print('try1')
                    sleep(3)
                    write_text = get_text(reg_url)
                except Exception:
                    try:
                        print('try2')
                        write_text = get_text(reg_url)
                    except Exception:
                        print('mai try law')
                        continue

            # print(write_text)
            if write_text is not None and len(write_text) > 0:
                F = open('%1d%1d%1d.txt' % (np.floor(counter/100), np.floor(counter/10), counter % 10), 'wt',
                         encoding='utf-8')
                F.writelines(write_text)
                F.close()
                sleep(0.2+0.1*np.random.rand())
            else:
                count += 1
                print('No text : %0f' % count)
    else:
        count += 1
        print('No File : %0f' % count)
    # print(os.getcwd())
    os.chdir('../..')


os.chdir('..')

