device = 'cuda'
train_path = 'data/new_train_all.txt' #'data/train.tsv'
dev_path = 'data/new_dev_all.txt'#'data/dev.tsv'
test_path = 'data/new_dev_all.txt'#'data/test.tsv'
max_vocab_size = 30000
embedding_size = 300

base_class = 'bert-base-cased-finetuned-mrpc'
tokenizer_class = 'bert-base-cased-finetuned-mrpc'

train_batch_size = 32
dev_batch_size = 512
test_batch_size = 500

max_grad_norm = 5.0
num_labels = 2
max_query_len = 64
l2_lambda = 0.000001
learning_rate = 3e-6
epochs = 100
mode = 'merged' #paws