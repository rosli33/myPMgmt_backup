import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

def train_ann_model():
    print("Loading data...")

    # Construct the correct file path
    data_path = 'C:\\Users\\CIKLEE\\Documents\\myPMgmt\\tasks\\model\\synthetic_data.csv'
    
    try:
        # Load the data
        tasks_df = pd.read_csv(data_path)

        # Print the column names to check
        print("Column names in the dataset:", tasks_df.columns)

        # Encode categorical variables
        label_encoders = {}
        for column in ['Task_Type', 'Current_Status', 'Deadline_Class', 'Business_Impact']:
            le = LabelEncoder()
            tasks_df[column] = le.fit_transform(tasks_df[column])
            label_encoders[column] = le

        # Feature Engineering
        tasks_df['Impact_Effort'] = tasks_df['Business_Impact'] * tasks_df['Days_to_Deadline']
        tasks_df['Days_to_Deadline'] = np.log1p(tasks_df['Days_to_Deadline'])  # Log transform for scaling

        # Select features for model training
        X = tasks_df[['Task_Type', 'Current_Status', 'Deadline_Class', 'Business_Impact', 'Days_to_Deadline']]
        y = tasks_df['Priority']

        # Encode the target variable (Priority)
        priority_le = LabelEncoder()
        y = priority_le.fit_transform(y)
        y_categorical = to_categorical(y)  # One-hot encode the target variable

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

        # Standardize input features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        y_train_classes = np.argmax(y_train, axis=1)  # Convert one-hot back to class labels
        X_train_balanced, y_train_balanced_classes = smote.fit_resample(X_train, y_train_classes)

        # One-hot encode the balanced target variable
        y_train_balanced = to_categorical(y_train_balanced_classes, num_classes=len(priority_le.classes_))

        # Build the optimized ANN model
        model = Sequential()
        model.add(Dense(128, input_shape=(X_train_balanced.shape[1],), activation='relu', kernel_regularizer=l1_l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(y_train_balanced.shape[1], activation='softmax'))  # Output layer

        # Compile the model with Adam optimizer
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Callbacks for optimization
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        # Train the model with callbacks
        model.fit(X_train_balanced, y_train_balanced, epochs=200, batch_size=64, validation_data=(X_test, y_test), 
                  callbacks=[early_stopping, lr_scheduler])

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Generate a classification report using numeric labels
        report = classification_report(y_test_classes, y_pred_classes)
        print(report)

        # Save the model
        model_dir = 'C:\\Users\\CIKLEE\\Documents\\myPMgmt\\tasks\\model\\'
        model_path = os.path.join(model_dir, 'final_ann_model.h5')
        model.save(model_path)
        print(f"Model trained and saved to {model_path} successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

train_ann_model()
