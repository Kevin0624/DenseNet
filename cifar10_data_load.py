import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def GetPhoto(pixel):
    assert len(pixel)==3072

    #對list進行切片操作，然後reshape
    r = pixel[0:1024].reshape(32, 32, 1)
    g = pixel[1024:2048].reshape(32, 32, 1)
    b = pixel[2048:3072].reshape(32, 32, 1)

    photo = np.concatenate([r, g, b], -1) # 把 R, G, B 疊在一起，變成 image

    return photo


# 根據想要的data 標籤 給 data 例如 : 要label 就會給 label ， 要 filename 就會給 filename
def GetTrainDataByLabel(label):
    batch_label = []  # 這一份batch 的名稱
    labels = []       # 圖片的 label
    data = []         # 圖片的 data
    filenames = []    # 圖片的 name

    for i in range(1, 6):
        batch_label.append(unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/data_batch_%d"%i)[b'batch_label'])
        labels += unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/data_batch_%d"%i)[b'labels']
        data.append(unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/data_batch_%d"%i)[b'data'])
        filenames += unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/data_batch_%d"%i)[b'filenames']

    data_array = np.array(data).reshape(50000, 3072)

    label = str.encode(label) # 

    if label == b'data':
        
        array = np.ndarray([len(data_array), 32, 32, 3], dtype=np.int32)
        for i in range(len(data_array)):
            
            array[i] = GetPhoto(data_array[i])
        
        return array

    elif label == b'labels':
        return labels

    elif label == b'batch_label':
        return batch_label
    
    elif label == b'filenames':
        return filenames
    
    else:
        raise NameError


def GetTestDataByLabel(label):
    

    batch_label = unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/test_batch")[b'batch_label']
    labels = unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/test_batch")[b'labels']
    data = unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/test_batch")[b'data']
    filenames = unpickle("C:/Users/WeiXiang/Documents/cifar10_data/cifar-10-batches-py/test_batch")[b'filenames']

    label = str.encode(label) # 

    if label == b'data':
        array = np.ndarray([len(data), 32, 32, 3], dtype=np.int32)
        for i in range(len(data)):
            array[i] = GetPhoto(data[i])
        
        return array

    elif label == b'labels':
        return labels

    elif label == b'batch_label':
        return batch_label
    
    elif label == b'filenames':
        return filenames
    
    else:
        raise NameError

    




    




