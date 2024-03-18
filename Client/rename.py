import os
path = "C:/Users/kj373/Documents/00-DATA/Experiment/azim_khan/qr/run_1/08_08_17_48_25/images"
for filename in os.listdir(path):
    num = filename[:-3]
    # num = num.zfill(4)
    # print(num)
    new_filename = num + ".jpg"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    print(new_filename)
