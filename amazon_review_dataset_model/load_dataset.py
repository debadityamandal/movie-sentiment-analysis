import bz2


def read_file(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode('utf-8')
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
    return texts, labels
