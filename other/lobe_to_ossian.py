import os

def lobe_to_ossian(in_dir):
    # convert a directory of .token files to .txt files
    for fname in os.listdir(in_dir):
        os.rename(os.path.join(in_dir,fname),
            os.path.join(in_dir, f"{os.path.splitext(fname)[0]}.txt"))

if __name__ == '__main__':
    lobe_to_ossian('corpus/is/speakers/margret_preamp/txt/')