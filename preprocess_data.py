import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
def add_noise_to_samples(sample, num_samples, noise_levels):
    sample_array = sample.to_numpy()
    new_samples = []
    for _ in range(num_samples):
        noise = np.random.normal(loc=0, scale=noise_levels, size=49)
        new_sample = sample_array + noise
        new_samples.append(new_sample)
    new_samples = np.array(new_samples)
    new_samples_df = pd.DataFrame(new_samples.reshape(num_samples, -1), columns=sample.columns)
    return new_samples_df

def create_numerics(data):
    nominal_cols = data.select_dtypes(include='object').columns.tolist()
    for nom in nominal_cols:
        enc = LabelEncoder()
        enc.fit(data[nom])
        data[nom] = enc.transform(data[nom])
    return data

def prepare_data(iter=42):
    data = pd.read_excel("Data/HEROdata2.xlsx")
    data = data.dropna()

    # Calculate noise levels as 5% of the standard deviation of each feature
    # noise_levels = data.std(numeric_only=True) * 0.05
    trojan_free = data[data['Label'] == "Trojan Free'"].reset_index(drop=True)
    # trojan_free.to_excel("C:/Users/Omee02/Desktop/Multi/codes/trojanfree.xlsx", index=False)
    for circuit in trojan_free['Circuit'].unique():
        circuit_data = data[data['Circuit'] == circuit]
        trojan_free_sample = circuit_data[circuit_data['Label'] == "Trojan Free'"]
        if len(trojan_free_sample) == 1:
            noise_levels = circuit_data.std(numeric_only=True)
            num_infected = len(circuit_data[circuit_data['Label'] != "Trojan Free'"])
            noise_samples = add_noise_to_samples(trojan_free_sample.drop(['Label', 'Circuit'], axis=1), num_samples=num_infected, noise_levels=noise_levels)
            noise_samples['Label'] = "Trojan Free'"
            noise_samples['Circuit'] = circuit
            data = pd.concat([data, noise_samples], ignore_index=True)

    data = create_numerics(data)
    data = data.sample(frac=1, random_state=iter).reset_index(drop=True)

    columns_to_drop = [
    'IO_Pad Internal Power',
    'IO_Pad Switching Power',
    'IO_Pad Leakage Power',
    'IO_Pad Total Power',
    'Memory Internal Power',
    'Memory Switching Power',
    'Memory Leakage Power',
    'Memory Total Power',
    'Black_Box Internal Power',
    'Black_Box Switching Power',
    'Black_Box Leakage Power',
    'Black_Box Total Power',
    'Number of macros/black boxes',
    'Macro/Black Box area',
    'Clock_Network Internal Power',
    'Clock_Network Leakage Power']
    data = data.drop(columns=columns_to_drop)

    labels = data['Label']
    data.drop(columns=['Label'], axis=1, inplace=True)
    
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    data_pca = pca.fit_transform(standardized_data)
    pca_data = pd.DataFrame(data_pca)
    pca_data['Label'] = labels
    pca_data.to_excel("C:/Users/Omee02/Desktop/Multi/codes/pca_data.xlsx", index=False)
    x = pca_data.drop(['Label'], axis=1)
    y = pca_data['Label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=iter)
    
    return x_train, x_test, y_train, y_test


def prepare_data_foc(iter=42):
    data = pd.read_excel("Data/HEROdata2.xlsx")
    data = data.dropna()

    # Calculate noise levels as 5% of the standard deviation of numeric features
    noise_levels = data.select_dtypes(include=[np.number]).std() * 0.05

    # Process each circuit family
    for circuit in data['Circuit'].unique():
        circuit_data = data[data['Circuit'] == circuit]
        trojan_free_data = circuit_data[circuit_data['Label'] == "Trojan Free"]

        if len(trojan_free_data) == 1:  # If only one Trojan Free sample in the family
            num_trojan = len(circuit_data) - len(trojan_free_data)
            noise_samples = add_noise_to_samples(trojan_free_data.drop(['Label', 'Circuit'], axis=1), 
                                                 num_samples=num_trojan, 
                                                 noise_levels=noise_levels.loc[trojan_free_data.drop(['Label', 'Circuit'], axis=1).columns])
            noise_samples['Label'] = "Trojan Free"
            noise_samples['Circuit'] = circuit
            data = pd.concat([data, noise_samples], ignore_index=True)

    # Convert categorical data to numeric
    data = create_numerics(data)  # Assuming a function that encodes nominal columns

    # Shuffle and reset index
    data = data.sample(frac=1, random_state=iter).reset_index(drop=True)

    # Drop unnecessary columns if any
    columns_to_drop = [
    'IO_Pad Internal Power',
    'IO_Pad Switching Power',
    'IO_Pad Leakage Power',
    'IO_Pad Total Power',
    'Memory Internal Power',
    'Memory Switching Power',
    'Memory Leakage Power',
    'Memory Total Power',
    'Black_Box Internal Power',
    'Black_Box Switching Power',
    'Black_Box Leakage Power',
    'Black_Box Total Power',
    'Number of macros/black boxes',
    'Macro/Black Box area',
    'Clock_Network Internal Power',
    'Clock_Network Leakage Power']
    data.drop(columns=columns_to_drop, inplace=True)

    # Standardize numeric features
    scaler = StandardScaler()
    features = data.select_dtypes(include=[np.number])
    scaled_features = scaler.fit_transform(features)
    
    # Optional PCA for dimensionality reduction
    pca = PCA(n_components=10)  # Adjust number of components as necessary
    data_pca = pca.fit_transform(scaled_features)
    pca_data = pd.DataFrame(data_pca)
    pca_data['Label'] = data['Label']


    # Splitting data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(pca_data.drop('Label', axis=1), pca_data['Label'], test_size=0.25, random_state=iter)

    return x_train, x_test, y_train, y_test
    