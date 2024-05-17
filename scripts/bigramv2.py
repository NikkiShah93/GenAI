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
head_size = 16
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

## the head class will use the self-attention concepts
class Head(nn.Module):
    def __init__(self, head_size=head_size, num_embd = num_embedding):
        super().__init__()
        ## we have a linear layer for keys
        ## one for queries and another for values
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.quey = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)
        ## we also have the triangle that we use for masking
        self.register_buffer(name='tril',tensor=torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape ## C = head_size
        k = self.key(x) ## (B, T, num_emb)
        q = self.quey(x) ## (B, T, num_emb)
        weight = q @ k.transpose(-2, -1) * C ** -0.5 ## (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        v = self.value(x) ## (B, T, C)
        out = weight @ v ## (B, T, C)
        return out

## next we have to build our model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size = vocab_size, num_emb=num_embedding, block_size=block_size):
        super().__init__()
        ## now we want to add two more layers
        ## and change our original embedding 
        ## to have vocab x num_emb shape
        self.embedding_table = nn.Embedding(vocab_size, num_emb)
        ## and then have another embedding for positions of the token
        self.pos_embedding_table = nn.Embedding(block_size, num_emb)
        ## we have to create a head in here now
        self.head = Head(head_size=num_emb, num_embd=num_emb)
        ## we also need a linear layer to go from token to logits
        self.lin_head = nn.Linear(num_emb, vocab_size)

    def forward(self, x, target=None):
        B, T = x.shape
        token_emb = self.embedding_table(x) ## (B, T, num_emb)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) ## (T, num_emb)
        ## and we add the emb for the identity of the token
        ## and its position in the block
        x = token_emb + pos_emb ## (B, T, num_emb)
        ## we have to do the Head's forward path on our x
        x = self.head(x)
        logits = self.lin_head(x) ## (B, T, vocab)
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
            cropped_x = x[:, -block_size:]
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
            _, loss = model(xs, ys)
            losses[i] = loss.item()
        out[split] = losses.mean()   
    model.train()     
    return out



## then we need to create an instance of the model
model = BigramLanguageModel(vocab_size=vocab_size).to(device)

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

