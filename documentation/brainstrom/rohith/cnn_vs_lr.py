import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import numpy as np

data = pd.read_csv('data_cleaning_games.csv', names=['sentence', 'label'])
test_data = pd.read_csv('confused_data.csv', names=['sentence', 'label'])
#print(data)
review = data['sentence'].values
label = data['label'].values
# split data into test and train
review_train, review_test, label_train, label_test = train_test_split(review, label, test_size=0.30, random_state=2000)

manual_test_sent = test_data['sentence'].values
manual_test_lab = test_data['label'].values

review_vectorizer = CountVectorizer()
review_vectorizer.fit(review_train)
Xlr_train = review_vectorizer.transform(review_train)
Xlr_test = review_vectorizer.transform(review_test)
Xlr_train
LRmodel = LogisticRegression()
LRmodel.fit(Xlr_train, label_train)
score = LRmodel.score(Xlr_test, label_test)
xlr_test_data = review_vectorizer.transform(manual_test_sent)
manual_score = LRmodel.score(xlr_test_data, manual_test_lab)
print("Accuracy:", score)
print("Manual Data Accuracy:", manual_score)

#CNN Implementation



tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)
Xcnn_train = tokenizer.texts_to_sequences(review_train)
Xcnn_test = tokenizer.texts_to_sequences(review_test)
Xcnn_test_data = tokenizer.texts_to_sequences(manual_test_sent)
vocab_size = len(tokenizer.word_index) + 1
print(review_train[1])
print(Xcnn_train[1])
maxlen = 100
Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
Xcnn_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)
Xcnn_test_data = pad_sequences(Xcnn_test_data, padding='post', maxlen=maxlen)
print(Xcnn_train[0, :])
embedding_dim = 200
textcnnmodel = Sequential()
textcnnmodel.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
textcnnmodel.add(layers.Conv1D(512, 5, activation='relu'))
textcnnmodel.add(layers.GlobalMaxPooling1D())
textcnnmodel.add(layers.Dense(15, activation='relu'))
textcnnmodel.add(layers.Dense(1, activation='sigmoid'))
textcnnmodel.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
textcnnmodel.summary()

textcnnmodel.fit(Xcnn_train, label_train,
                     epochs=50,
                     verbose=False,
                     validation_data=(Xcnn_test, label_test),
                     batch_size=20)
loss, accuracy = textcnnmodel.evaluate(Xcnn_train, label_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = textcnnmodel.evaluate(Xcnn_test, label_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = textcnnmodel.evaluate(Xcnn_test_data, manual_test_lab, verbose=False)
print("Manual Testing Accuracy:  {:.4f}".format(accuracy))
