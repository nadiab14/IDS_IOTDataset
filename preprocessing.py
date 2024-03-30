import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def normalization(df, col):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(-1, 1))
    return df

def preprocess_data(filepath,test_size):
    # Load the data from CSV
    data_types = {'icmp.ident': str}
    data = pd.read_csv(filepath , dtype=data_types)
    print("*********************data************")
    print( data.head(1))



    # Display header
    #print("Dataset Header:")
    #print(data.head())


    # Display the updated columns and missing values distribution
    #print("Updated Columns:")
    #print(data.columns)

    #print("\nMissing values distribution after dropping columns:")
    #print(data.isnull().mean())

    # check datatype in each column
    #print("Column datatypes: ")
    #print(data.dtypes)

    data.drop_duplicates(inplace=True)


    # Display the content of object-type columns
    object_columns = data.select_dtypes(include='object').columns
   # print(data[object_columns].head())

    # Convert hexadecimal strings to numeric values for 'ip.flags'
    data['ip.flags'] = data['ip.flags'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

    # Convert hexadecimal strings to numeric values for 'tcp.flags'
    data['tcp.flags'] = data['tcp.flags'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)



    """colonnes_a_supprimer = ['icmp.seq', 'icmp.seq_le', 'ip.checksum.status', 'tcp.checksum.status',
                            'frame.time_delta',
                            'frame.time_delta_displayed', 'frame.time_epoch', 'frame.time_relative', 'ip.ttl',
                            'tcp.time_relative',
                            'udp.srcport', 'udp.dstport', ]
    # Supprimez les colonnes spécifiées
    data.drop(colonnes_a_supprimer, axis=1, inplace=True)"""

      # Set the threshold for missing values (e.g., 70%)
    """threshold = 0.8

    # Calculate the proportion of missing values in each column
    missing_values_proportion = data.isnull().mean()

    # Get the columns with missing values above the threshold
    columns_to_drop = missing_values_proportion[missing_values_proportion > threshold].index

    # Drop the columns with high levels of missing data
    data = data.drop(columns=columns_to_drop)"""

    # Convert columns with numeric values to the appropriate data types
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)

    # Forward fill missing values ,remplir en prenant la valeur de la ligne précédente dans le même ensemble de données
    #data.fillna(method='ffill', inplace=True)
    #remplir les colonnes vides avec0
    #data.fillna(0, inplace=True)
    #remplir les colonnes vides avec mean

    # Extract the 'label' column from the original data
    label = data['label']

    # Drop 'attack_cat' and 'label' columns
    data = data.drop(columns=['attack_cat', 'label'])
    #print("***************************************",data.shape[1])
    # Normalize the numeric columns using the normalization function
    num_col = list(data.columns)
    data.fillna(0, inplace=True)

    #data.fillna(data.mean(), inplace=True)
    #data.fillna(0, inplace=True)
    data_normalized = normalization(data.copy(), num_col)
    print("*********************data normalaized************")
    print(data_normalized.head(1))

    # Add back the 'label' column to the normalized DataFrame
    data_normalized['label'] = label

    # Display the updated DataFrame after normalization
    #print("Updated DataFrame after normalization:")
    #print(data_normalized.head())

    # Convert the data into a numpy array
    data_numeric = data_normalized[num_col]
    data_array = data_numeric.to_numpy()
    #print("Dataset Header:")
    #print(data_numeric.head())

    # Display unique values for each column
    """print("\nUnique Values in Each Column:")
    for col in data_numeric.columns:
        #print(f"{col}: {data_numeric[col].value_counts()}")
        print(f"{col}: {data_numeric[col].nunique()}")"""

    # Plot the pie chart for the binary labels
    """plt.figure(figsize=(8, 8))
    plt.pie(data_normalized['label'].value_counts(), labels=['normal', 'abnormal'], autopct='%0.2f%%')
    plt.title("Pie chart distribution of normal and abnormal labels", fontsize=16)
    plt.legend()
    plt.savefig('C:\\Users\\Olfa\\PycharmProjects\\codeArticle2\\Pie_chart_binary.png')
    #plt.show()
    """
    # Step 1: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_array, label, test_size=0.3, random_state=1, stratify=label)

    return X_train, X_test, y_train, y_test, label
