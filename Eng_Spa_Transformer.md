
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
