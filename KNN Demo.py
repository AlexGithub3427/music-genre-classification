import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# loading flattened training data
X_train = np.load("flattened_X_train.npy")
y_train = np.load("flattened_y_train.npy")

# load flattened test data
X_test = np.load("X_test_full.npy")
y_test = np.load("y_test_full.npy")

# fitting the label encoder on training labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# initializing knn with best parameters determined (not loading model since this takes longer, just manually defining)
knn = KNeighborsClassifier(
    n_neighbors=20,
    weights="distance",
    metric="manhattan"
)

# fitting model on training data
knn.fit(X_train, y_train_enc)

# generating encoded predictions on test set
y_pred_enc = knn.predict(X_test)

# converting predictions back to labels (genres)
y_pred_labels = le.inverse_transform(y_pred_enc)

# calculating probability of each genre for the prediction
probs = knn.predict_proba(X_test)

# prints the true genre
print(y_pred_labels)

# prints the predicted genre
print(y_test)

# making a df of the probabilities for each genre, then printing to see the confidence for each genre
df_probs = pd.DataFrame({
    "genre": le.classes_,
    "probability": probs[0]
})

# printing the probability table
print(df_probs)