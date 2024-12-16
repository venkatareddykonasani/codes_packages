### Import Statements and Initial Configuration

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
```

- **`os` Module**: Provides a way to interact with the operating system.
- **Set `KERAS_BACKEND`**: Ensures that TensorFlow is used as the backend for Keras. This is important when running the code in environments where multiple backends might be available.

---

```python
import pathlib
import random
import string
import re
import numpy as np
```

- **`pathlib`**: Simplifies file path manipulation in a platform-independent way.
- **`random` and `string`**: Used for generating random data and working with string operations.
- **`re`**: Provides regular expression functionality for pattern matching and text processing.
- **`numpy` (`np`)**: Fundamental library for numerical computations, often used for handling arrays, matrices, and numerical operations.

---

```python
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
```

- **TensorFlow Imports**:
  - **`tensorflow.data` (`tf_data`)**: For working with TensorFlow datasets, enabling efficient data input pipelines.
  - **`tensorflow.strings` (`tf_strings`)**: Facilitates string operations within TensorFlow workflows.
- **Keras Imports**:
  - **`layers`**: Includes prebuilt layers for constructing neural networks.
  - **`ops`**: Likely an abstraction for TensorFlow operations in newer versions of Keras.
  - **`TextVectorization`**: A layer to preprocess and vectorize text data for NLP tasks.

---

### Custom Transformer Package

```python
# The below package is a custom transformer package stored on my GitHub
!wget https://raw.githubusercontent.com/venkatareddykonasani/codes_packages/refs/heads/main/transformer.py
import transformer
from transformer import TransformerEncoder, TransformerDecoder, PositionalEmbedding
```

- **Download Custom Package**: A custom Python file, `transformer.py`, is downloaded from a GitHub repository using the `wget` command.
  - **Note**: Running this command requires internet access and the `wget` utility.
- **Import Custom Components**:
  - **`TransformerEncoder`**: Encodes input sequences into a meaningful representation.
  - **`TransformerDecoder`**: Decodes the encoded representation into output sequences.
  - **`PositionalEmbedding`**: Adds positional context to input embeddings, a crucial component of Transformer models.

---

### Explanation of Dependencies

1. **External Libraries**:
   - **TensorFlow & Keras**: Core libraries for machine learning and neural networks.
   - **NumPy**: Provides efficient numerical computation.
2. **Custom Code**:
   - A custom transformer implementation is sourced externally, and key classes (`TransformerEncoder`, `TransformerDecoder`, `PositionalEmbedding`) are imported.

---
### Downloading the Data

```python
datafile_location = "https://github.com/venkatareddykonasani/Datasets/raw/Spa_Eng_Dataset/spa-eng.zip"
# One more option
# datafile_location = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
```

- **Dataset URL**: The dataset is hosted at two possible locations:
  - **GitHub**: Custom repository for datasets.
  - **Google Cloud Storage**: Alternative source often used by TensorFlow.

```python
eng_spa_raw = keras.utils.get_file(
    fname="spa-eng.zip",
    origin=datafile_location,
    extract=True,
)
```

- **Download and Extract**: 
  - `keras.utils.get_file` downloads the file from `datafile_location`.
  - The file is automatically extracted (`extract=True`).
  - `fname` specifies the local filename for the downloaded file.

```python
eng_spa_raw = pathlib.Path(eng_spa_raw).parent / "spa-eng" / "spa.txt"
```

- **File Path Adjustment**: Constructs the path to the extracted `spa.txt` file, where the actual data resides.

---

### Parsing the Data

```python
with open(eng_spa_raw) as f:
    lines = f.read().split("\n")[:-1]
```

- **Load File**: Opens the file and reads its content.
- **Split Lines**: Splits the data into individual lines. The `[:-1]` removes any trailing empty line.

```python
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))
```

- **Extract Sentence Pairs**:
  - Each line contains an English sentence (source) and its corresponding Spanish sentence (target), separated by a tab (`\t`).
  - `[start]` and `[end]` tokens are appended to the Spanish sentence to mark the beginning and end of the sequence for the model.
- **Store Pairs**: The English and Spanish sentence pairs are stored as tuples in the `text_pairs` list.

```python
print("Sample Data Points")
for i in random.sample(range(1, 100000), 5):
    print(text_pairs[i])
```

- **Display Random Samples**:
  - Randomly selects 5 indices from a range (1 to 100,000).
  - Prints the English-Spanish sentence pairs at those indices.

---

### Splitting the Data into Train, Validation, and Test Sets

```python
random.shuffle(text_pairs)
```

- **Shuffle Data**: Randomizes the order of the sentence pairs to ensure that training, validation, and test sets are well-distributed.

```python
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
```

- **Calculate Splits**:
  - 15% of the data is reserved for validation.
  - Another 15% is reserved for testing.
  - The remaining data (70%) is used for training.

```python
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]
```

- **Split Data**:
  - `train_pairs`: Contains the first `num_train_samples` pairs.
  - `val_pairs`: Contains the next `num_val_samples` pairs.
  - `test_pairs`: Contains the remaining `num_val_samples` pairs.

```python
print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

- **Display Split Counts**: Prints the total number of sentence pairs and their distribution across training, validation, and test sets.

---

### Key Points

1. **Dataset Source**:
   - The dataset is downloaded from a predefined URL and automatically extracted.
   - It contains parallel English-Spanish sentence pairs.

2. **Preprocessing**:
   - Sentences are tokenized by splitting on tab characters.
   - `[start]` and `[end]` tokens are added to the target (Spanish) sentences.

3. **Data Splits**:
   - Training set (70%), validation set (15%), and test set (15%) are created after shuffling the data.

4. **Example Usage**:
   ```python
   print(train_pairs[0])  # Prints the first training pair.
   ```

---

### Example Output (Sample Execution)

```plaintext
Sample Data Points
('I want to sleep.', '[start] Quiero dormir. [end]')
('I have a big dog.', '[start] Tengo un perro grande. [end]')
...

117914 total pairs
82539 training pairs
17687 validation pairs
17688 test pairs
```

#### Overview of `TextVectorization`

The `TextVectorization` layer transforms text data into integer sequences:
- **Integer Sequences**: Each word in the text is mapped to an integer, which represents its index in a predefined vocabulary.
- **Two Instances**: One layer is used for English text, and another is used for Spanish text, with different configurations.

---

#### Configuration Parameters

```python
vocab_size = 15000
sequence_length = 20
batch_size = 64
```

- **`vocab_size`**: Maximum number of words in the vocabulary. Words outside the top 15,000 most frequent ones are replaced with an "out of vocabulary" (OOV) token.
- **`sequence_length`**: Maximum length of sequences. Longer sequences are truncated, and shorter ones are padded.
- **`batch_size`**: The number of samples per batch for processing. This parameter is used later when creating datasets.

---

#### Handling Custom Standardization for Spanish Text

```python
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
```

- **Define Characters to Strip**: A set of characters (`strip_chars`) is defined, which includes all punctuation plus the Spanish-specific character `¿`.
- **Exclude Square Brackets**: `[` and `]` are excluded from the set to avoid conflicts with other text-processing steps.

```python
def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
```

- **Custom Standardization**:
  - Converts input strings to lowercase.
  - Removes all characters defined in `strip_chars` using TensorFlow's `regex_replace`.

---

#### Create Text Vectorization Layers

```python
eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
```

- **English Vectorization**:
  - Default standardization (lowercase, punctuation stripping).
  - Outputs sequences of integers with a fixed length of 20.

```python
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
```

- **Spanish Vectorization**:
  - Applies `custom_standardization`.
  - Outputs sequences of integers with a length of 21 (`sequence_length + 1`) to account for `[start]` and `[end]` tokens.

---

#### Preparing Training Text Data

```python
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
```

- **Extract English and Spanish Text**: 
  - `train_eng_texts`: List of English sentences.
  - `train_spa_texts`: List of Spanish sentences, including `[start]` and `[end]` tokens.

---

#### Adapting Vectorization Layers

```python
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)
```

- **Adapt Vocabulary**:
  - Learns the vocabulary and word frequencies from the training data.
  - Maps each word to an integer index.

---

#### Print Example Vectorized Text

```python
for i in random.sample(range(1, 100000), 3):
    print(train_eng_texts[i], '\n', eng_vectorization(train_eng_texts[i]))
    print(train_spa_texts[i], '\n', spa_vectorization(train_spa_texts[i]))
    print("=============")
```

- **Random Samples**:
  - Selects three random indices from the training data.
  - Prints the raw English and Spanish sentences alongside their vectorized representations.

---

### Example Output

For an English-Spanish pair:

```plaintext
Original English Sentence:
"I want to sleep."

Vectorized English Sentence:
[11, 85, 3, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Original Spanish Sentence:
"[start] Quiero dormir. [end]"

Vectorized Spanish Sentence:
[1, 34, 56, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
=============
```

- **English Sentence**: Vectorized into integers with padding (`0`) to ensure the sequence length is 20.
- **Spanish Sentence**: Includes `[start]` (`1`) and `[end]` (`2`) tokens, vectorized into integers with padding to ensure the sequence length is 21.

---

### Key Points

1. **Custom Standardization**: Tailors text preprocessing for Spanish by handling specific characters like `¿`.
2. **Fixed Sequence Length**: Ensures uniform input size for the model by truncating or padding sequences.
3. **Vocabulary**: Learns and limits the vocabulary to the top `vocab_size` most frequent words.
4. **Compatibility**: Prepares the text data for input to deep learning models.
### Explanation of Data Preprocessing Code

This section focuses on preparing the dataset for training a sequence-to-sequence model. The preprocessing aligns with the input requirements of the model and ensures efficient data handling.

---

### Key Objective

At each training step:
- The model predicts the next word(s) in the target sequence (`targets`) using:
  - The full source sequence (`encoder_inputs`).
  - The partial target sequence so far (`decoder_inputs`).

---

#### **`format_dataset` Function**

```python
def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )
```

1. **Vectorization**:
   - Converts `eng` (English) and `spa` (Spanish) sentences into integer sequences using the respective `TextVectorization` layers.

2. **Structure of Input and Output**:
   - **`encoder_inputs`**: Full vectorized source sentence (`eng`).
   - **`decoder_inputs`**: Target sentence excluding the last word (`spa[:, :-1]`). This is used to predict the next word in the target sequence.
   - **`targets`**: Target sentence excluding the first word (`spa[:, 1:]`), which is the expected output for the model.

---

#### **`make_dataset` Function**

```python
def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)
```

1. **Input Texts**:
   - Extracts English (`eng_texts`) and Spanish (`spa_texts`) sentences from the paired data.

2. **Create Dataset**:
   - **`tf_data.Dataset.from_tensor_slices`**: Creates a TensorFlow dataset from the English and Spanish text lists.
   - **`batch(batch_size)`**: Batches the data into groups of size `batch_size` (64 here).

3. **Mapping with `format_dataset`**:
   - Transforms each batch of English and Spanish sentences into the desired input-output format using `format_dataset`.

4. **Dataset Optimization**:
   - **`cache()`**: Caches the dataset in memory for faster access during training.
   - **`shuffle(2048)`**: Randomizes the order of data points for better generalization.
   - **`prefetch(16)`**: Prefetches 16 batches of data asynchronously to ensure smooth data loading during training.

---

#### Creating Training and Validation Datasets

```python
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
```

- **`train_ds`**: TensorFlow dataset for the training split.
- **`val_ds`**: TensorFlow dataset for the validation split.

---

#### Inspecting the Dataset

```python
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")
```

1. **Iterate Over Batches**:
   - `train_ds.take(1)` retrieves the first batch of data.

2. **Print Shapes**:
   - **`inputs["encoder_inputs"].shape`**: Shape of the source sequence, expected to be `(batch_size, sequence_length)`.
   - **`inputs["decoder_inputs"].shape`**: Shape of the partial target sequence (`decoder_inputs`), expected to be `(batch_size, sequence_length)`.
   - **`targets.shape`**: Shape of the target sequence (`targets`), expected to be `(batch_size, sequence_length)`.

---

### Expected Shapes

Given:
- `batch_size = 64`
- `sequence_length = 20` (for English) and `21` (for Spanish, due to `[start]` and `[end]` tokens):

Output shapes for a batch:
```plaintext
inputs["encoder_inputs"].shape: (64, 20)
inputs["decoder_inputs"].shape: (64, 20)
targets.shape: (64, 20)
```

---

### Key Points

1. **Dynamic Input and Output**:
   - `encoder_inputs`: Full source sentence.
   - `decoder_inputs`: Partial target sentence up to the current prediction step.
   - `targets`: Next word(s) to predict in the target sentence.

2. **Dataset Efficiency**:
   - The `cache`, `shuffle`, and `prefetch` methods optimize the dataset pipeline for training.

3. **Batch Processing**:
   - Processes data in batches, with uniform sequence lengths achieved through padding or truncation.

4. **Alignment with Model Input Requirements**:
   - Ensures compatibility with the sequence-to-sequence model, where the encoder-decoder framework expects input-output pairs in this format.
  
### Explanation of Model Configuration Code

This section defines the Transformer-based encoder-decoder model for sequence-to-sequence tasks, such as machine translation. The model uses components like positional embeddings, transformer encoder/decoder layers, and a dense output layer.

---

### Key Hyperparameters

```python
embed_dim = 256
latent_dim = 2048
num_heads = 8
```

- **`embed_dim`**: Dimensionality of the token and positional embeddings. It defines the size of the vector representations for each word.
- **`latent_dim`**: Size of the feed-forward layer within each transformer block. Larger dimensions improve model capacity but require more computation.
- **`num_heads`**: Number of attention heads in the multi-head self-attention mechanism. This allows the model to capture different aspects of the input data.

---

### Encoder Configuration

#### Input Layer

```python
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
```

- **Shape**: `(batch_size, sequence_length)`, where `sequence_length` can vary (`None`).
- **Data Type**: Integer indices representing words in the vocabulary.
- **Name**: `encoder_inputs` identifies this input for the encoder model.

#### Positional Embedding

```python
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
```

- **`PositionalEmbedding`**: Adds positional information to token embeddings to handle word order, as Transformers are order-agnostic.
- **Input**: Integer token indices from `encoder_inputs`.
- **Output**: A tensor of shape `(batch_size, sequence_length, embed_dim)`.

#### Transformer Encoder

```python
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
```

- **`TransformerEncoder`**: Processes the embedded input using:
  - **Self-attention**: Captures relationships between words in the input sequence.
  - **Feed-forward layers**: Adds non-linearity and increases representation capacity.
- **Output**: Encoded representation of the input sequence, used as input for the decoder.

#### Encoder Model

```python
encoder = keras.Model(encoder_inputs, encoder_outputs)
```

- **Input**: Tokenized source sentence (`encoder_inputs`).
- **Output**: Encoded representation (`encoder_outputs`).

---

### Decoder Configuration

#### Input Layers

```python
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
```

- **`decoder_inputs`**: Integer token indices for the target sentence so far (used to predict the next token).
- **`encoded_seq_inputs`**: Encoded output from the encoder, providing context to the decoder.

#### Positional Embedding

```python
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
```

- **Functionality**: Adds positional information to the decoder's input tokens.

#### Transformer Decoder

```python
x = TransformerDecoder(embed_dim, latent_dim, num_heads)([x, encoder_outputs])
```

- **`TransformerDecoder`**:
  - **Masked self-attention**: Ensures that the decoder can only attend to earlier positions in the sequence.
  - **Encoder-decoder attention**: Attends to the encoder's output (`encoder_outputs`), integrating information from the source sequence.

---

#### Dropout and Dense Layer

```python
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
```

- **Dropout**: Prevents overfitting by randomly setting some elements to zero during training.
- **Dense Layer**:
  - Maps the decoder's output to the vocabulary size (`vocab_size`).
  - Uses `softmax` to produce probabilities for each word in the vocabulary.

#### Decoder Model

```python
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
```

- **Inputs**: 
  - `decoder_inputs`: Partial target sequence.
  - `encoded_seq_inputs`: Context from the encoder.
- **Output**: Predicted probabilities for the next word in the target sequence.

---

### Combined Transformer Model

```python
transformer = keras.Model(
    {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
    decoder_outputs,
    name="transformer",
)
```

- **Inputs**: 
  - `encoder_inputs`: Source sentence.
  - `decoder_inputs`: Partial target sentence.
- **Output**: Predicted probabilities for the next word in the target sentence.
- **Name**: The combined model is named `transformer`.

---

### Summary of the Transformer Model

1. **Encoder**:
   - Encodes the source sequence into a latent representation.
   - Captures relationships between tokens in the input sentence.

2. **Decoder**:
   - Generates the target sequence one word at a time.
   - Uses:
     - Its own past outputs (`decoder_inputs`).
     - Context from the encoder (`encoded_seq_inputs`).

3. **Combined Model**:
   - Encapsulates the encoder-decoder relationship.
   - Processes input-output pairs for sequence-to-sequence tasks.

---

### Example Output Shapes

Assuming:
- **Batch Size**: `64`
- **Sequence Length**: `20`
- **Embedding Dim**: `256`
- **Vocab Size**: `15000`

- **Encoder Outputs**:
  - Shape: `(64, 20, 256)` (batch size, sequence length, embedding dim).
- **Decoder Outputs**:
  - Shape: `(64, 20, 15000)` (batch size, sequence length, vocab size).
 
### Explanation of the Remaining Code

This final part covers training, saving/loading model weights, and making predictions using the Transformer model. 

---

### Model Summary and Compilation

```python
transformer.summary()
```

- **`summary()`**: Prints a summary of the model architecture, including layer details, output shapes, and parameter counts.

```python
transformer.compile(
    "rmsprop",
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"],
)
```

- **Optimizer**: `rmsprop` is used to optimize model parameters during training.
- **Loss Function**: `SparseCategoricalCrossentropy` is applied, with `ignore_class=0` to avoid penalizing padding tokens during training.
- **Metric**: Accuracy is used for monitoring training progress, but this is a basic metric for machine translation.

---

### Model Training

```python
epochs = 1  # This should be at least 30 for convergence
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
```

- **`fit()`**: Trains the model using the training dataset (`train_ds`) and evaluates on the validation dataset (`val_ds`).
- **Epochs**: Set to `1` for quick demonstration but should ideally be increased to `30` or more for convergence.

---

### Saving and Loading Model Weights

#### Save Model Weights

```python
transformer.save_weights("eng_spa_weights.h5")
```

- Saves the trained model weights to a file (`eng_spa_weights.h5`) for reuse.

#### Load Model Weights

```python
transformer.load_weights("eng_spa_weights.h5")
```

- Loads the previously saved model weights to resume training or make predictions.

#### Download Pretrained Weights

```python
!gdown 'https://drive.google.com/uc?export=download&id=1jMLFnlXPQXlRRVmXfr2sIOuxUO4VGuKo' -O eng_spa_50epochs.weights.h5
```

- Downloads pretrained weights (`eng_spa_50epochs.weights.h5`) for a model trained over 50 epochs.

#### Fine-tuning with Pretrained Weights

```python
transformer.load_weights("eng_spa_50epochs.weights.h5")
transformer.fit(train_ds, epochs=1, validation_data=val_ds)
```

- After loading the pretrained weights, the model is fine-tuned with one additional epoch using the training and validation datasets.

---

### Making Predictions

#### Spanish Vocabulary and Index Lookup

```python
spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
```

- **`spa_vocab`**: Retrieves the Spanish vocabulary from the `spa_vectorization` layer.
- **`spa_index_lookup`**: Creates a dictionary to map integer indices back to their corresponding Spanish tokens.

#### Decode Sequence Function

```python
def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer(
            {
                "encoder_inputs": tokenized_input_sentence,
                "decoder_inputs": tokenized_target_sentence,
            }
        )

        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence
```

1. **Input Sentence Tokenization**:
   - Converts the input English sentence into vectorized tokens using `eng_vectorization`.

2. **Decoding Loop**:
   - **Initialization**: Start decoding with the token `"[start]"`.
   - **Step-by-Step Prediction**:
     - Vectorize the partially decoded target sentence.
     - Use the `transformer` model to predict the next token's probabilities.
     - Choose the token with the highest probability (`argmax`).
   - **Update Decoded Sentence**: Append the predicted token to the decoded sentence.

3. **Stopping Condition**:
   - Break the loop if the `"[end]"` token is generated or if the `max_decoded_sentence_length` is reached.

---

### Prediction Examples

#### Translate a Sample Sentence

```python
input_sentence = "A nearly perfect photo"
translated = decode_sequence(input_sentence)
print(input_sentence, " ==> ", translated)
```

- Translates a single English sentence into Spanish using the `decode_sequence` function.

#### Random Test Sentences

```python
test_eng_texts = [pair[0] for pair in test_pairs]
for i in random.sample(range(1, 100000), 5):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
```

- Randomly selects English sentences from the test set (`test_eng_texts`) and translates them into Spanish.

---

### Key Points

1. **Training**:
   - Training for just 1 epoch will not lead to convergence; at least 30 epochs are recommended.
   - Pretrained weights can accelerate convergence and improve accuracy.

2. **Prediction**:
   - The decoder generates one word at a time, appending to the target sequence until the `"[end]"` token is reached.
   - Predictions are based on both the encoder's context and the partially decoded sequence.

3. **Metrics**:
   - Accuracy is used for simplicity, but metrics like BLEU are more suitable for machine translation tasks.

4. **Efficiency**:
   - Leveraging pre-trained weights (`50 epochs`) ensures higher-quality translations without requiring extensive computation from scratch.
