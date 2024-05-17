## in this version we want use the self-attention as well

## first the imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path 

## general params
PATH_DIR = Path('../data/')
PATH_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = 'wizardOfOz.txt'
FILE_PATH = PATH_DIR / FILE_NAME

## the main hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-2
evaluation_interval = 300
epochs = 3000
num_embedding = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## then the same script that was developed in the notebook

## reading the file
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

## next we want to create our vocab
vocab = sorted(set(text))
vocab_size = len(vocab)

## and then the mapping and decoding/encoding functions
string_to_int = {s:i for i, s in enumerate(vocab)}
int_to_string = {i:s for i, s in enumerate(vocab)}

## and then the decoder/encode
encode = lambda x:[string_to_int[l] for l in x]
decode = lambda x:''.join([int_to_string[i] for i in x])

## next we want to create a tensor from our encoded text
text_tensor = torch.tensor(encode(text), dtype=torch.long)

## and split the tensor into train and test
train, test = np.split(text_tensor, [int(len(text_tensor)*.8)])
print(f'train set shape: {train.shape}, test set shape: {test.shape}')
## and then we need a function to split the data into batches
def get_batch(split):
    data = train if split == 'train' else test
    random_batch_id = torch.randint(len(data)-block_size, size=(batch_size,))
    X = torch.stack([data[i:i+block_size] for i in random_batch_id])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_batch_id])
    X, y = X.to(device), y.to(device)
    return X, y
X_train, y_train = get_batch('train')
print('The training set shapes: ',X_train.shape, y_train.shape)
X_test, y_test = get_batch('test')
print('The test set shapes: ',X_test.shape, y_test.shape)

## next we have to build our model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        ## now we want to add two more layers
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, target=None):
        logits = self.embedding_table(x)
        if target is None:
            loss = None
        else:
            ## we have to change the shape for loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, x, num_max_iter):
        for _ in range(num_max_iter):
            logits, loss = self(x)
            logits = logits[:,-1, :]
            probs = torch.softmax(logits)
            next_x = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_x), dim=1)
        return x

## we also need a function to calculate the average loss
## we need to put the model into eval mode
## to turn off the gradient calculation
@torch.inference_mode()
def estimate_loss(evaluation_interval):
    out = {}
    for split in ['train', 'test']:
        losses = torch.zeros(evaluation_interval)        
        for i in range(evaluation_interval):
            xs, ys = get_batch(split)
            model.eval()
            logits, loss = model(xs, ys)
            losses[i] = loss.item()
        out[split] = losses.mean()   
    model.train()     
    return out

## then we need to create an instance of the model
model = BigramLanguageModel(vocab_size).to(device)

## and then create an instance of the optimizer
optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = learning_rate)

## and finally the training loop
for e in range(epochs):
    xs, ys = get_batch('train')
    logits, loss = model(xs, ys)
    ## zero out the gradient
    optimizer.zero_grad(set_to_none=True)
    ## and then backpropagation
    loss.backward()
    ## and then the optimizer step
    optimizer.step()
    if e%evaluation_interval==0:
        result = estimate_loss(evaluation_interval)
        print(f"Epoch {e} average train loss is {result['train']:.2f} | average test loss is {result['test']:.2f}")

