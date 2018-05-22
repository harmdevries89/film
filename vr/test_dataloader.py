import time
import os

from vr.data import ClevrDataLoader

loader_kwargs = {
    'question_h5': os.path.join(os.environ['data_dir'], 'train_questions.h5'),
    'feature_h5': os.path.join(os.environ['data_dir'], 'train_features.h5'),
    'vocab': os.path.join(os.environ['data_dir'], 'vocab.json'),
    'batch_size': 64,
    'shuffle': 0,
  }

loader = ClevrDataLoader(**loader_kwargs)

t = time.time()
i = 0
for batch in loader:
    if i > 100:
        break
    i += 1
    print(i)
print(time.time() - t)
