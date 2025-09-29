# General Training Controls

This training setup consists of 4 components. The main script in the pyscripts folder that loads the model, dataset and optimizer, and the training parameters like learning rate, number of epochs, etc. The trainer.py script in the common folder holds the training loop. Changes can be made here when new techniques like learning rate scheduler need to be added. The distasync.py and reducer.py contain the DDAP wrapper and the communication related code respectively. These can largely be left untouched. One will primarily need to create a new main script to run this setup. Ensure that common folder is in the PYTHONPATH environment variable

```python
#Set the seed for deterministic executions if needed 
seed = 2354234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

#Initialize Process Group
init_process_group(backend="nccl",timeout=delta)

#Setup Training Parameters
config = Config(dataloaderprocs=5,batch_size=16,epochs=6,learning_rate=5e-5)

#Load the Model and its tokenizer from online or from disk
model_name = "xxx"
model_conf = ConfType.from_pretrained(model_name)
model = ModelType(config=model_conf)
tokenizert = TokenizerType.from_prtrained(model_name)
#ModelType, TokenizerType and ConfType will generally be available from HuggingFace

#Load The optimizer
optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

#Load The Dataset
#Either get it online from HuggingFace and tokenize (the tokenize_function might require some changes based on model and dataset)
train_dataset = load_dataset_split("=DatasetName", "DatasetSubname", split='train')
test_dataset = load_dataset_split("=DatasetName", "DatasetSubname", split='test')
def tokenize_function(examples):
   return tokenizer(
       examples["text"],
       truncation=True,
       padding=True,
       max_length=512,
   )
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=32)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=32)

#At this point you can save it to disk using dataset.save_to_disk("location") if needed to avoid tokenizing every execution

#Otherwise if the tokenized dataset is already in disk
tokenized_train_dataset = load_from_disk("location")
tokenized_test_dataset = load_from_disk("location")

#If training on a partition instead of the entire dataset
tokenized_train_dataset = tokenized_train_dataset.select(range(int(ratio*len(tokenized_train_dataset)))) #Set ratio as needed

#Set up a data collator if needed for cases like MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

#Pass everything to the Trainer
trainer = Trainer(config,model,optimizer,tokenizer=tokenizer,train_dataset=tokenized_train_dataset,test_dataset=tokenized_test_dataset, data_collator=data_collator, seed=seed, useDDAP=True)
#By Default useDDAP is False. Set it to True to use the DistributedDataAsynchronousParallel module for asynchronous training. Otherwise the synchronous training using DDP will happen.

#Set up any timers to measure time here

trainer.train()

#Measure time here

destroy_process_group()
```

This is a generic main script. You can refer to DDPbertc4.py and DDAPbertc4.py as reference and compare to this to get an idea of how to make any script for this setup. In general, switching between DDAP and DDP will only involving setting the useDDAP argument passed to the Trainer as True or False.
