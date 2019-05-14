import os
from time import sleep,time
import deepcut
import pickle
import numpy as np
import datetime

owd = os.getcwd()
print(owd)


def extract_word():
    os.chdir('data_dek-d_I')
    list_dir1 = os.listdir()

    # writer
    if list_dir1 is not None:
        for ii in range(2475, len(list_dir1)):
            the_dir1 = list_dir1[ii]

            time_now = datetime.datetime.now().strftime('%Y%m%d-%H:%M')
            print(time_now)

            print(ii, the_dir1)
            os.chdir(the_dir1)
            list_dir2 = os.listdir()

            # novel
            if list_dir2 is not None and list_dir2 != []:
                for the_dir2 in list_dir2:
                    os.chdir(the_dir2)
                    list_dir3 = os.listdir()

                    # chapter
                    if list_dir3 is not None and list_dir3 != []:
                        doc = ''
                        token_all = []
                        count_chap = 1

                        for the_dir3 in list_dir3:
                            file = the_dir3
                            token = ''
                            novel = ''

                            f = open(file, "r", encoding="utf-8")
                            novel = f.read()
                            token = deepcut.tokenize(novel)

                            doc += novel
                            token_all += token

                            if len(token_all) >= 2000:

                                file_name = (the_dir1 +'_'+ the_dir2 + '_%1d%1d%1d.txt' % (np.floor(count_chap / 100), np.floor(count_chap/10) % 10, count_chap % 10))

                                F = open('C:\\Users\First ThinkPad\Desktop\Thai Stylometry\data_raheem\document\\'+
                                         str(file_name), 'wt', encoding='utf-8')
                                F.writelines(doc)
                                F.close()

                                # save token file
                                with open('C:\\Users\First ThinkPad\Desktop\Thai Stylometry\data_raheem\\token\\' +
                                          str(file_name), "wb") as fp:  # Pickling
                                    pickle.dump(token_all, fp, protocol=pickle.HIGHEST_PROTOCOL)

                                count_chap += 1

                                token_all = []
                                doc = ''

                        if count_chap < 5 and count_chap > 1:

                            for each_chap in range(1, count_chap):
                                file_name = (the_dir1 + '_' + the_dir2 + '_%1d%1d%1d.txt' % (
                                    np.floor(each_chap / 100), np.floor(each_chap / 10) % 10, each_chap % 10))
                                # delete one by one
                                os.remove('C:\\Users\First ThinkPad\Desktop\Thai Stylometry\data_raheem\document\\'+ str(file_name))
                                os.remove(
                                    'C:\\Users\First ThinkPad\Desktop\Thai Stylometry\data_raheem\\token\\' + str(
                                        file_name))

                    os.chdir('..')

                os.chdir('..')


if __name__ == '__main__':
    extract_word()

# def into_dir(list_dir):
#     if list_dir is not None:
#         for the_dir in list_dir:
#             os.chdir(the_dir)
#             list_dir = os.listdir()
#
#     return list_dir
#
