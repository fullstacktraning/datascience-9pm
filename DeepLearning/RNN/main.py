import numpy as np

# Sequential - used to build layer by layer
from tensorflow.keras.models import Sequential
# Embedding - converts words to number vectors
# Ex. [0 0 3] ---> [0.00123, 0.00456,0.00432,.......]
# SimpleRNN - heart of RNN, remember previous words
# Dense - output layer
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
# convert statements to words
# hello how are you (tokens)
from tensorflow.keras.preprocessing.text import Tokenizer
# RNN, expecting equal length
# Ex1.  Hello 0  0.      Ex2.How are you ---> How - 1. are - 2. you - 3 
from tensorflow.keras.preprocessing.sequence import pad_sequences


english_sentences = [
    "hello",            
    "how are you",      
    "good morning",     
    "thank you",        
    "good night"        
]

telugu_sentences = [
    "నమస్కారం",
    "మీరు ఎలా ఉన్నారు",
    "శుభోదయం",
    "ధన్యవాదాలు",
    "శుభ రాత్రి"
]

# Step - 1
# create object to Tokenizer 
tokenizer = Tokenizer()

# Training
tokenizer.fit_on_texts(english_sentences)  

# Tokenization
input_sequences = tokenizer.texts_to_sequences(english_sentences)

print("Tokenized Sequences:")
print(input_sequences)


max_length = max(len(seq) for seq in input_sequences)

# print("Max Length:", max_length)



input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_length
)

print("After Padding:")
print(input_sequences)


# Assign number to each Telugu sentence
output_labels = np.array([1, 2, 3, 4, 5])

# Telugu dictionary mapping
telugu_mapping = {
    1: "నమస్కారం",
    2: "మీరు ఎలా ఉన్నారు",
    3: "శుభోదయం",
    4: "ధన్యవాదాలు",
    5: "శుభ రాత్రి"
}

# ============================================
# Step 6: Build RNN Model
# ============================================

model = Sequential()

# Embedding Layer
model.add(
    Embedding(
        input_dim=50,
        output_dim=8,                 
        input_length=max_length
    )
)

# RNN Layer
model.add(SimpleRNN(16))  # 16 - medium. 4/8 -- small.  64 -- strong.  128 -- too strong

# Output Layer
model.add(Dense(6, activation='softmax'))

# ============================================
# Step 7: Compile Model
# ============================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# Step 8: Train Model
# ============================================

model.fit(
    input_sequences,
    output_labels,
    epochs=500,
    verbose=1
)

#================================
# save the model
#================================
model.save("translator_model.h5")


# ============================================
# Step 9: Test Translator
# ============================================
test_sentence = ["hello"]

# Convert to sequence
test_seq = tokenizer.texts_to_sequences(test_sentence)

# Padding
test_seq = pad_sequences(test_seq, maxlen=max_length)

# Predict
prediction = model.predict(test_seq)

# Get highest probability index
predicted_class = np.argmax(prediction)

# ============================================
# Step 10: Print Telugu Output
# ============================================

print("\nEnglish Input:", test_sentence[0])

print("Predicted Telugu Output:")

print(telugu_mapping[predicted_class])

# ============================================
# Optional Probability Output
# ============================================

print("\nPrediction Probabilities:")
print(prediction)