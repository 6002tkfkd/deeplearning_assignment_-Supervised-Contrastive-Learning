from datasets import load_dataset
import pickle, os, numpy as np

save_dir = '/workspace/SupCon/data/cifar-10-batches-py'
os.makedirs(save_dir, exist_ok=True)

print('CIFAR-10 다운로드 중...')
ds = load_dataset('cifar10')

def save_batch(split_data, path):
    data = np.array([np.array(x['img']) for x in split_data])  # (N, 32, 32, 3)
    data = data.transpose(0, 3, 1, 2).reshape(len(data), -1)   # (N, 3072)
    labels = [x['label'] for x in split_data]
    with open(path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

print('train 배치 저장 중...')
train = list(ds['train'])
for i in range(5):
    chunk = train[i*10000:(i+1)*10000]
    save_batch(chunk, f'{save_dir}/data_batch_{i+1}')
    print(f'  data_batch_{i+1} 저장 완료')

print('test 배치 저장 중...')
save_batch(list(ds['test']), f'{save_dir}/test_batch')

meta = {'label_names': ['airplane','automobile','bird','cat','deer',
                        'dog','frog','horse','ship','truck']}
with open(f'{save_dir}/batches.meta', 'wb') as f:
    pickle.dump(meta, f)

print('Done! CIFAR-10 준비 완료')
