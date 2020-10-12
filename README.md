# Text-classification-arxiv-articles

MulNB_NumpyOnly.py is implemented from scratch to train a Naive Bayes classifier for text classification.

To run this code, you need to get enough corpus files which includes a "train.csv", a "test.csv" and a "stopwords.txt". All files are provided in this depository and for the last file you can also change it into yours. The depository also provide an example for submission file and we will get one after we run this code.

About the corpus input, you just need to change each input file's relative path at the main() function which locates at the last line. Here, you can simply make the file and the code under the same root and make the path as "train.csv", "test.csv".

About the result output, you just need to input the write out file path at the main() as the mention above. The output file e.g. "sub_NumpyOnly.csv" is the file to submit to the leadboard. Meanwhile, you will get the print type result of the accuracy at training set and validation set.
