import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import jax.numpy as jnp
from torchvision import datasets, transforms

from scipy.io import arff
def generate_dataset_semeion_pca_from_file(n_components=8, test_size=0.15, val_size=0.15, random_state=42, arff_file="phptd5jYj.arff"):
    """
    Loads the Semeion dataset from an ARFF file, applies PCA to reduce it to n_components,
    scales the PCA features to [0, 1], and splits the data into training, validation, and test sets.
    
    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing.
                           If set to 0, no test split is created.
      val_size (float): Fraction of the (train+validation) split to use as validation.
                        If set to 0, no validation split is created.
      random_state (int): Random seed for reproducibility.
      arff_file (str): Filename or path to the ARFF file.
      
    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling.
                        Returns an empty array if val_size==0.
      y_val (jnp.array): Validation set labels.
                        Returns an empty array if val_size==0.
      X_test (jnp.array): Test set features after PCA and scaling.
                        Returns an empty array if test_size==0.
      y_test (jnp.array): Test set labels.
                        Returns an empty array if test_size==0.
    """
    # Load ARFF file using scipy.io.arff
    data_arff, meta = arff.loadarff(arff_file)
    df = pd.DataFrame(data_arff)
    
    # The ARFF file contains 256 feature columns and one label column (named "Class").
    # In your printed output, the features appear as V1, V2, ..., V256 and the label column as "Class".
    feature_cols = [f"V{i}" for i in range(1, 257)]
    label_col = "Class"
    
    # Convert the features. (They should be numeric; if they are bytes, convert them to float.)
    data = df[feature_cols].values.astype(np.float32)
    
    # The Class column might be stored as bytes; convert it to integer labels.
    if df[label_col].dtype == object:
        labels = df[label_col].apply(lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x)).values.astype(np.int32)
    else:
        labels = df[label_col].values.astype(np.int32)
    
    # First split: separate out the test set if test_size > 0.
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val, y_train_val = data, labels
        X_test = np.empty((0, data.shape[1]), dtype=np.float32)
        y_test = np.empty((0,), dtype=labels.dtype)
    
    # Second split: separate training and validation sets if val_size > 0.
    if val_size > 0.0:
        # When test_size > 0, X_train_val is 1 - test_size fraction of data;
        # the effective validation size is adjusted accordingly.
        test_size_val = val_size if test_size == 0.0 else val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]), dtype=np.float32)
        y_val = np.empty((0,), dtype=y_train_val.dtype)
    
    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val) if X_val.shape[0] > 0 else np.empty((0, n_components), dtype=np.float32)
    X_test_pca = pca.transform(X_test) if X_test.shape[0] > 0 else np.empty((0, n_components), dtype=np.float32)
    
    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca) if X_val_pca.shape[0] > 0 else X_val_pca
    X_test_scaled = scaler.transform(X_test_pca) if X_test_pca.shape[0] > 0 else X_test_pca
    
    # Convert the numpy arrays to JAX arrays.
    return (
        jnp.array(X_train_scaled, dtype=jnp.float32),
        jnp.array(y_train, dtype=jnp.int32),
        jnp.array(X_val_scaled, dtype=jnp.float32),
        jnp.array(y_val, dtype=jnp.int32),
        jnp.array(X_test_scaled, dtype=jnp.float32),
        jnp.array(y_test, dtype=jnp.int32)
    )


def generate_dataset_semeion_pca(n_components=8, test_size=0.15, val_size=0.15, random_state=42):
    """
    Loads the Semeion dataset from the UCI repository, applies PCA to reduce to n_components, 
    scales the PCA features to [0, 1], and splits the data into training, validation, and test sets.
    Returns the datasets as JAX arrays.

    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing.
                           If set to 0, no test split is created.
      val_size (float): Fraction of the (train+validation) split to use as validation.
                        If set to 0, no validation split is created.
      random_state (int): Random seed for reproducibility.

    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling.
                        Returns an empty array if val_size is 0.
      y_val (jnp.array): Validation set labels.
                        Returns an empty array if val_size is 0.
      X_test (jnp.array): Test set features after PCA and scaling.
                        Returns an empty array if test_size is 0.
      y_test (jnp.array): Test set labels.
                        Returns an empty array if test_size is 0.
    """
    # Download the Semeion dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
    df = pd.read_csv(url, sep='\s+', header=None)

    # The dataset has 256 feature columns (pixel values) and 10 columns for one-hot encoded labels
    data = df.iloc[:, :256].values
    labels_onehot = df.iloc[:, 256:].values

    # Convert one-hot encoded labels to single integer labels
    labels = np.argmax(labels_onehot, axis=1)

    # First split: (Train+Validation) and Test.
    if test_size > 0.0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val, y_train_val = data, labels
        X_test = np.empty((0, data.shape[1]))
        y_test = np.empty((0,), dtype=labels.dtype)

    # Second split: Training and Validation.
    if val_size > 0.0:
        # If test_size==0, the fraction for validation is just val_size.
        test_size_val = val_size if test_size == 0.0 else val_size/(1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, X_train_val.shape[1]))
        y_val = np.empty((0,), dtype=y_train_val.dtype)

    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    # Only apply PCA transform on test set if it has samples.
    if X_test.shape[0] > 0:
        X_test_pca = pca.transform(X_test)
    else:
        X_test_pca = np.empty((0, n_components))

    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca)
    # Only transform test set if non-empty.
    if X_test_pca.shape[0] > 0:
        X_test_scaled = scaler.transform(X_test_pca)
    else:
        X_test_scaled = np.empty((0, n_components))

    # Convert the numpy arrays to JAX arrays.
    return (jnp.array(X_train_scaled, jnp.float32), jnp.array(y_train, jnp.int32),
            jnp.array(X_val_scaled, jnp.float32), jnp.array(y_val, jnp.int32),
            jnp.array(X_test_scaled, jnp.float32), jnp.array(y_test, jnp.int32))


def generate_dataset_fashion_mnist_pca(n_components=8, test_size=0.2, val_size=0.2, random_state=42):
    """
    Loads Fashion-MNIST, applies PCA to reduce to n_components, scales the PCA features to [0, 1],
    and splits the data into training, validation, and test sets. Supports test_size or val_size = 0.
    
    Parameters:
      n_components (int): Number of PCA components/features.
      test_size (float): Fraction of the full dataset to reserve for testing. If 0, no test set is created.
      val_size (float): Fraction of the (train+val) split to use as validation. If 0, no validation set is created.
      random_state (int): Random seed for reproducibility.
      
    Returns:
      X_train (jnp.array): Training set features after PCA and scaling.
      y_train (jnp.array): Training set labels.
      X_val (jnp.array): Validation set features after PCA and scaling (empty if val_size==0).
      y_val (jnp.array): Validation set labels (empty if val_size==0).
      X_test (jnp.array): Test set features after PCA and scaling (empty if test_size==0).
      y_test (jnp.array): Test set labels (empty if test_size==0).
    """
    import numpy as np
    from torchvision import datasets, transforms
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    import jax.numpy as jnp

    # Define a transform to convert images to flattened tensors.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor (values in [0, 1])
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image to a vector
    ])
    
    # Load Fashion-MNIST datasets.
    fashion_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract data and labels as numpy arrays.
    X_train_full = np.array([np.array(img) for img, _ in fashion_train])
    y_train_full = np.array([label for _, label in fashion_train])
    X_test_full = np.array([np.array(img) for img, _ in fashion_test])
    y_test_full = np.array([label for _, label in fashion_test])
    
    # Combine train and test sets for a full dataset.
    dataset = np.concatenate([X_train_full, X_test_full], axis=0)
    labels = np.concatenate([y_train_full, y_test_full], axis=0)
    
    # First split: Separate test set if test_size > 0.
    if test_size > 0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            dataset, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_val = dataset
        y_train_val = labels
        X_test = np.empty((0, dataset.shape[1]))
        y_test = np.empty((0,), dtype=labels.dtype)
    
    # Second split: Create validation set if val_size > 0.
    if val_size > 0:
        test_size_for_val = val_size / (1 - test_size) if test_size > 0 else val_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_for_val, random_state=random_state, stratify=y_train_val
        )
    else:
        X_train = X_train_val
        X_val = np.empty((0, X_train_val.shape[1]))
        y_val = np.empty((0,), dtype=y_train_val.dtype)
    
    # Apply PCA on the training data only.
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val) if X_val.shape[0] > 0 else X_val
    X_test_pca = pca.transform(X_test) if X_test.shape[0] > 0 else X_test
    
    # Scale the PCA features to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_val_scaled = scaler.transform(X_val_pca) if X_val_pca.shape[0] > 0 else X_val_pca
    X_test_scaled = scaler.transform(X_test_pca) if X_test_pca.shape[0] > 0 else X_test_pca
    
    # Convert the numpy arrays to JAX arrays.
    return (jnp.array(X_train_scaled, dtype=jnp.float32), jnp.array(y_train, dtype=jnp.int32),
            jnp.array(X_val_scaled, dtype=jnp.float32), jnp.array(y_val, dtype=jnp.int32),
            jnp.array(X_test_scaled, dtype=jnp.float32), jnp.array(y_test, dtype=jnp.int32))