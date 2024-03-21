from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from datetime import datetime


df = pd.read_csv('file.csv')

import pandas as pd
import random
from datetime import datetime, timedelta


def generate_features(df):
    # List of date formats
    date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%y-%m-%d', '%d-%m-%y', '%m-%d-%y']
    # Generate 1000 random dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2030, 12, 31)
    dates = [(start_date + (end_date - start_date) * random.random()).date() for _ in range(1000)]
    # Format dates and append to dataframe
    for date in dates:
        format = random.choice(date_formats)  # Choose a random date format
        feature = date.strftime(format) + '.pdf'  # Append '.pdf' to the file name
        df.loc[len(df)] = [feature, 0]  # Append new row to dataframe
    return df


# Use the function
df2 = pd.DataFrame(columns=['feature', 'label'])
df2 = generate_features(df2)

df = pd.concat([df, df2], ignore_index=True)

# Define the alphabet and the input and output sizes
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
input_size = 100
output_size = 14

# Create a tokenizer that converts characters to indices
tokenizer = Tokenizer(num_words=alphabet_size + 1, char_level=True)
tokenizer.fit_on_texts(alphabet)

X = tokenizer.texts_to_sequences(df['feature'])
# Pad or truncate the rows to have the same length
X = pad_sequences(X, maxlen=input_size)
# Convert the label column to a binary vector
y = df['label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_val = X_val.astype('float32')
y_val = y_val.astype('float32')

# LSTM Model parameters
lstm_units = 64  # Number of LSTM units in the LSTM layer

# Define the model parameters
vocab_size = 100  # The size of the vocabulary
max_len = 100  # The maximum length of the input sequence
embed_dim = 10  # The dimension of the embedding
num_filters = 2  # The number of filters for the convolutional layer
kernel_size = 12  # Reduced kernel size
pool_size = 2  # Pool size for MaxPooling1D
hidden_dim1 = 32  # The dimension of the hidden layer
hidden_dim2 = 10  # The dimension of the hidden layer


# Define the LSTM model
model = Sequential()
model.add(Embedding(alphabet_size + 1, embed_dim, input_length=max_len))
model.add(LSTM(lstm_units, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(lstm_units))
model.add(Dropout(0.2))
model.add(Dense(hidden_dim1, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_dim2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('LSTM_model_1.h5', verbose=1, monitor='val_loss',
                             save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])


import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model


df = pd.read_csv('file.csv')
df['feature'] = df['feature'].str.replace(' ', '').str.lower()
input_size = 100
X_test_file = pad_sequences(tokenizer.texts_to_sequences(df['feature']), maxlen=input_size)
y_test = df['label']

X_test_file = X_test_file.astype('float32')
y_test = y_test.astype('float32')

arr = model.evaluate(X_test_file, y_test)

loss = arr[0]
accuracy = arr[1]

# Save results to a file
with open('LSTM_model_1_loss_acc.txt', 'w') as file:
    file.write(f'Loss: {loss}\n')
    file.write(f'Accuracy: {accuracy}\n')

# Save the tokenizer
with open('LSTM_model_1_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('LSTM_model_1_history.csv')

# Save the model
model.save('LSTM_model_1.h5')


'''

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
with open('LSTM_model_1_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained LSTM model
model = load_model('LSTM_model_1.h5')

# The input_size should be the same as what was used during training
input_size = 100

# Function to preprocess new features
def preprocess_features(features, tokenizer, input_size):
    # Lowercase and remove spaces
    features = [name.replace(' ', '').lower() for name in features]
    # Convert features to sequences
    sequences = tokenizer.texts_to_sequences(features)
    # Pad or truncate the sequences to have the same length
    padded_sequences = pad_sequences(sequences, maxlen=input_size)
    return padded_sequences

# Function to make predictions on new features
def predict_new_features(features, tokenizer, model, input_size):
    # Preprocess the features
    X_new = preprocess_features(features, tokenizer, input_size)
    # Make predictions
    predictions = model.predict(X_new)
    return predictions

# Example of using the functions with new features
new_features = ['feature']
predicted_classes = predict_new_features(new_features, tokenizer, model, input_size)

# Output the predictions
for feature, predicted_class in zip(new_features, predicted_classes):
    print(f"The file '{feature}' is predicted as class {predicted_class[0]}")


'''