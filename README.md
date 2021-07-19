# Danish Text Classification with BERT
Text classification of actual multi-label and multi-class [covid-19 questions in Danish labeled by BotXO](https://github.com/certainlyio/corona_dataset).

## Loading COVID-19 dataset
```python
# Read .CSV dataset
ds_covid = 'path/to/data_covid19/danish.csv'
df_covid = pd.read_csv(ds_covid)
df_covid.columns = ['Language', 'Domain', 'Intent', 'Industry', 'Text', 'Want To']
data = df_covid[['Domain', 'Intent', 'Text']]

# Filtering rows
data = data.groupby('Text').filter(lambda x : len(x) < 3) # drops comments with less than 3 characters

# Categorize fine-tuning data labels
data['Domain_label'] = pd.Categorical(data['Domain'])
data['Intent_label'] = pd.Categorical(data['Intent'])

# Transforms labels to numerical values
data['Domain'] = data['Domain_label'].cat.codes
data['Intent'] = data['Intent_label'].cat.codes
data.Domain.value_counts()
data.Intent.value_counts()

# Save copy of entire dataset
entire_dataset = data.copy() 

# Split data into fine-tuning and test samples
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['Intent']], random_state=123)
print('Train data shapes:', data.shape)
print('Test data shapes:', data_test.shape)
```

## Train and test dataset shapes
```python
# Train data shapes: (764, 5)
# Test data shapes: (191, 5)
```

## Load and train the model
```python
# Load pre-trained Danish BERT model from BotXO and build multiclass classification model with Keras
folder_bert = '/path/to/bert-base-danish'

# Fine-tune different versions of model on different number of epochs
epochs = [5,10,15,20,25,30]

# Create an empty test results dataframe
results = pd.DataFrame(index=range(len(epochs)), columns=['Epochs','Domain (F1)', 'Domain (acc)','Intent (F1)', 'Intent (acc)'])
results['Epochs'] = epochs

for i, epchs in enumerate(epochs):

  # Config loaded with output_hidden_states set to False
  config = BertConfig.from_pretrained(folder_bert + '/bert_config.json')
  config.output_hidden_states = False
  print('1. BERT Config loaded.')

  # BERT tokenizer loaded
  tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = folder_bert, config = config)
  print('2. BERT Tokenizer loaded.')
  # Transformers BERT model loaded
  tf_BERT_model = TFBertModel.from_pretrained(folder_bert, from_pt=True, config = config)
  print('3. BERT Model loaded.')

  # Max token length
  max_length = 100

  # The MainLayer of BERT is loaded
  BERT = tf_BERT_model.layers[0]

  # Construct model inputs, attention masking is included
  input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
  attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
  inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
  #inputs = {'input_ids': input_ids}

  # The BERT model is loaded as a single layer into the Keras modelling
  BERT_model = BERT(inputs)[1]
  dropout = Dropout(config.hidden_dropout_prob, name='pooledOutput')
  pooledOutput = dropout(BERT_model, training=False)

  # Predictive outputs for the 'Domain' and 'Intent' labels of the COVID-19 dataset are constructed
  domain = Dense(units=len(data.Domain_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='domain')(pooledOutput)
  intent = Dense(units=len(data.Intent_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='intent')(pooledOutput)
  outputs = {'domain': domain, 'intent': intent}

  # The above is put into the Model() function to create a Keras model object 
  model = Model(inputs=inputs, outputs=outputs, name='Danish_BERT_MultiClass_Model')
  #model.summary()

  # Model fine-tuning:
  # Adam optimization algorithm with gradient clipping
  opt = Adam(learning_rate=5e-05,epsilon=1e-08,decay=0.01,clipnorm=1.0)

  # Use categorical losses and metrics
  loss = {'domain': CategoricalCrossentropy(from_logits = True), 'intent': CategoricalCrossentropy(from_logits = True)}
  metric = {'domain': CategoricalAccuracy('accuracy'), 'intent': CategoricalAccuracy('accuracy')}

  # Tested some countermeasures for overfitting
  #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
  #mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

  # Compiling
  model.compile(optimizer = opt, loss = loss, metrics = metric)

  # Preprocess labels into categories
  y_domain = to_categorical(data['Domain'])
  y_intent = to_categorical(data['Intent'])

  # Apply tokenizer to text/question inputs from users in the dataset
  x = tokenizer(
      text=data['Text'].to_list(),
      add_special_tokens=True,
      max_length=max_length,
      truncation=True,
      padding='max_length', # Instead of equal to True, the input sequences are padded all the way up to max_length such that all inputs are of the same length
      return_tensors='tf',
      return_token_type_ids = False,
      return_attention_mask = True, 
      verbose = True)

  # Model is fit to the fine-tuning data with a small validation split
  history = model.fit(
      x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
      #x={'input_ids': x['input_ids']},
      y={'domain': y_domain, 'intent': y_intent},
      validation_split=0.2,
      batch_size=32, # 64 gives OOM errors
      epochs=epchs) # Loops over different number of epochs
      #callbacks=[es,mc])
  model.save('/path/to/models/model_'+str(epchs)+'_epchs')

  # Preprocess the test data labels into categorical
  test_y_domain = to_categorical(data_test['Domain'])
  test_y_intent = to_categorical(data_test['Intent'])

  test_x = tokenizer(
      text=data_test['Text'].to_list(),
      add_special_tokens=True,
      max_length=max_length,
      truncation=True,
      padding='max_length',
      return_tensors='tf',
      return_token_type_ids = False,
      return_attention_mask = True,
      verbose = True)

  '''
  model_evaluation = model.evaluate(
      #x={'input_ids': test_x['input_ids']},
      x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']},
      y={'issue': test_y_domain, 'product': test_y_intent}
  )
  '''

  # Predictions & model evaluation
  preds = model.predict( x={'input_ids': test_x['input_ids'], 'attention_mask': test_x['attention_mask']}) 

  # Taking argmax to set highest value category equal to 1, rest equal to 0.
  predsDomain = np.zeros_like(preds['domain'])
  predsDomain[np.arange(len(preds['domain'])), preds['domain'].argmax(1)] = 1 #argmax of highest value at set that value equal to 1, the rest of the classes are 0
  predsIntent = np.zeros_like(preds['intent'])
  predsIntent[np.arange(len(preds['intent'])), preds['intent'].argmax(1)] = 1

  # Evaluation scores
  print('F1 Score of Domain:', round(f1_score(test_y_domain, predsDomain, average='weighted'),3))
  print('Accuracy Score of Domain: ', round(accuracy_score(test_y_domain, predsDomain),3))
  print('F1 Score of Intent:', round(f1_score(test_y_intent, predsIntent, average='weighted'),3))
  print('Accuracy Score of Intent: ', round(accuracy_score(test_y_intent, predsIntent),3))

  # Test data results tabel
  results = pd.read_csv('path/to/results.csv') # Read current results table
  results['Domain (F1)'].loc[i] = round(f1_score(test_y_domain, predsDomain, average='weighted'),3)
  results['Domain (acc)'].loc[i] = round(accuracy_score(test_y_domain, predsDomain),3)
  results['Intent (F1)'].loc[i] = round(f1_score(test_y_intent, predsIntent, average='weighted'),3)
  results['Intent (acc)'].loc[i] = round(accuracy_score(test_y_intent, predsIntent),3) 
  results.to_csv('path/to/results.csv',index=False)
  results.to_latex('path/toresults.tex',index=False)

# Save fine-tuning history of model_30_epochs
json.dump(history.history, open('path/to/model_30_epochs_history.json', 'w'))
```
## Training history on labels of 'Domain' and 'Intent'
![Training history](https://github.com/jacobshein/Danish-Text-Classification-with-BERT/blob/main/history.png)

## Test results benchmarked to logit model with L1 regularization 
|  Model    |   Domain (F1) |   Domain (acc) |   Intent (F1) |   Intent (acc) |
| :---------|--------------:|---------------:|--------------:|---------------:|
|  Logit L1 |         0.906 |          0.906 |         0.702 |          0.712 |
|  5 epchs  |         0.907 |          0.916 |         0.378 |          0.424 |
|  10 epchs |         0.931 |          0.932 |         0.534 |          0.592 |
|  15 epchs |         0.936 |          0.937 |         0.721 |          0.749 |
|  20 epchs |         0.938 |          0.937 |         0.764 |          0.785 |
|  25 epchs |         0.963 |          0.963 |         0.761 |          0.780 |
|  30 epchs |         0.969 |          0.969 |         0.809 |          0.827 |
```python

```

## Predictions the model got wrong
|  Text                                                            | Domain Label                | Prediction                  |
| :----------------------------------------------------------------|:----------------------------|:----------------------------|
|  Har de fundet en vaccine?                                       | About Coronavirus           | Coronavirus Recommendations |
|  Hvad indeholder regeringens hjælpepakke?                        | About Coronavirus           | Coronavirus Recommendations |
|  Har du kontakt information til callcenter                       | Contact                     | Coronavirus Recommendations |
|  I hvilke tilfælde af utilpashed bør man opsøge læge             | Coronavirus Recommendations | About Coronavirus           |
|  kan man blive smittet i svæmmehallen                            | Coronavirus Recommendations | About Coronavirus           |
|  Bliver børn egentlig smittet?                                   | Coronavirus Recommendations | About Coronavirus           |
|  hvad er det styrelsen beder os om                               | Coronavirus Recommendations | About Coronavirus           |
|  hvad skal jeg gøre som arrangør, når vores event bliver afholdt | Gatherings                  | Coronavirus Recommendations |
|  Telefonhjælp corona                                             | Contact                     | About Coronavirus           |
|  er der nogen der bliver lettere smittet end andre               | Coronavirus Recommendations | About Coronavirus           |
|  Kan jeg sende alle mine medarbejdere hjem?                      | Coronavirus Recommendations | Employer Guidelines         |
|  hvad gør i på nuværende tidspunkt                               | About Coronavirus           | Coronavirus Recommendations |
|  er der kommet nogen opdateringer for nyligt                     | About Coronavirus           | Coronavirus Recommendations |


### Dataset

* Pre-trained BERT was fine-tuned on [real COVID-19 questions in Danish](https://github.com/certainlyio/corona_dataset).

### Contact

* For information on usage, fine-tuning procedure and more, please reach out on email through [jacobhein.com](https://jacobhein.com/#contact).

## Reference

* [Pre-trained model by BotXO](https://github.com/botxo/nordic_bert)

