from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from scipy.optimize import root
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
global ann_acc,mlr_accuracy

global filename
global df, X_train, X_test, y_train, y_test



main = tk.Tk()
main.title("Machine Learning based Rainfall Prediction ") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = Label(main, text='Machine Learning based Rainfall Prediction ',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)


font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

# Upload Function
def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")


font1 = ('times', 13, 'bold')
# Upload Crop Recommendation button with color sky blue
uploadButton = Button(main, text="Upload Dataset", command=upload, bg="sky blue")
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=330, y=550)

# Function to open the graph
def open_graph():
    fig = plt.figure(figsize=(20, 5))
    ax = df['RainTomorrow'].value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'navy'], alpha=0.9, rot=0)
    plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.show()

# Create a button to open the graph
open_graph_button = tk.Button(main, text="Open Graph", command=open_graph, bg="light blue")
open_graph_button.place(x=950, y=550)
open_graph_button.config(font=font1)

import seaborn as sns

# Function to handle null values and display heatmap
def handle_nulls():
    plt.figure(figsize=(20, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='PuBu')
    plt.title('Heatmap of Null Values')
    plt.show()

# Create a button to handle null values and display heatmap
handle_nulls_button = tk.Button(main, text="Handle Nulls", command=handle_nulls, bg="light green")
handle_nulls_button.place(x=1100, y=550)
handle_nulls_button.config(font=font1)

# Prediction function
def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s",(X_test[i], y_pred[i]))
    return y_pred

# Split the dataset
def splitdataset(): 
    global df, X_train, X_test, y_train, y_test
    
    # Fill missing values with the mean of each column
    df.fillna(df.mean(), inplace=True)
    
    # Encode categorical variables to integers
    label_encoder = LabelEncoder()
    df['Date'] = label_encoder.fit_transform(df['Date'])  # Assuming 'Date' is categorical
    
    X = df[['Date', 'MinTemp', 'MaxTemp', 'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']]

    y = df['RainTomorrow']
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n\n")
    text.insert(END, str(X))
    text.insert(END, str(y))
    
    return X, y, X_train, X_test, y_train, y_test


# Split Dataset button with color light green
splitButton = Button(main, text="Split Dataset", command=splitdataset, bg="light green")
splitButton.place(x=50, y=650)
splitButton.config(font=font1)





def ann():
    global X_train, X_test, y_train, y_test, ann_acc
    text.delete('1.0', END)
    
    # Initialize and train the ANN model
    ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    ann_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = ann_model.predict(X_test)
    
    # Calculate accuracy
    ann_acc = accuracy_score(y_test, y_pred)
    
    # Display accuracy in the text widget
    text.insert(END, "Prediction Results (Artificial Neural Network)\n\n")
    text.insert(END, f"Accuracy of ANN: {ann_acc * 100:.2f}%\n\n")

# Decision Tree button with color turquoise
dtButton = Button(main, text="Run ANN", command=ann, bg="turquoise",width=14)
dtButton.place(x=190, y=650)
dtButton.config(font=font1)



def mlr():
    global X_train, X_test, y_train, y_test
    global mlr_accuracy,mlr_model
    
    # Initialize and train the MLR model
    mlr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    mlr_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = mlr_model.predict(X_test)
    
    # Calculate accuracy
    mlr_accuracy = accuracy_score(y_test, y_pred)
    
    # Display accuracy in the text widget
    text.delete('1.0', END)
    text.insert(END, "Prediction Results (Multinomial Logistic Regression)\n\n")
    text.insert(END, f"Accuracy of MLR: {mlr_accuracy * 100:.2f}%\n\n")

# RF Algorithm button with color coral
ranButton = Button(main, text="Run MLR Algorithm", command=mlr, bg="coral")
ranButton.place(x=350,y=650)
ranButton.config(font=font1)

#Lodistic Regression Function
def logisticRegression():
    global lr_accuracy
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    text.delete('1.0', END)
    text.insert(END,"Prediction Results\n\n")
    y_pred = model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy logistic regression: {lr_accuracy * 100}")
    text.insert(END,"Accuracy logistic regression : "+str(lr_accuracy*100)+"\n\n")

# Logistic Regression button with color gold
LRButton = Button(main, text="Logistic Regression", command=logisticRegression, bg="gold")
LRButton.place(x=530,y=650)
LRButton.config(font=font1)



import matplotlib.pyplot as plt


# Global variables
global lr_accuracy

# Other global variables and functions...
import matplotlib.pyplot as plt

def accuracy_graph():
    global lr_accuracy, ann_acc, mlr_accuracy

    # Check if accuracies are available
    if lr_accuracy is None or ann_acc is None or mlr_accuracy is None:
        print("Please run Logistic Regression, ANN, and MLR algorithms first to calculate accuracies.")
        return

    algorithms = ['Logistic Regression', 'Artificial Neural Network', 'MLR']
    accuracy_scores = [lr_accuracy * 100, ann_acc * 100, mlr_accuracy * 100]

    # Bar graph
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, accuracy_scores, color=['blue', 'green', 'orange'])
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 100)

    # Add text labels on top of the bars
    for i in range(len(algorithms)):
        plt.text(i, accuracy_scores[i], f'{accuracy_scores[i]:.2f}%', ha='center', va='bottom')

    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(accuracy_scores, labels=algorithms, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Accuracy Comparison')

    plt.tight_layout()
    plt.show()


# Show Accuracy Graph button with color violet
accuracy_graph_button = Button(main, text="Show Accuracy Graph", command=accuracy_graph, bg="violet",width=17)
accuracy_graph_button.place(x=720, y=650)
accuracy_graph_button.config(font=font1)
    





def predict():
    global mlr_model
    if mlr_model is None:
        messagebox.showerror("Error", "Please train the MLR model first.")
        return
    
    # Open file manager for selecting CSV file
    file_path = filedialog.askopenfilename(initialdir="dataset", title="Select CSV file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    
    if not file_path:
        return
    
    print("File selected:", file_path)
    
    try:
        # Read the selected CSV file
        new_df = pd.read_csv(file_path)
        
        # Assuming the structure of new_df is similar to the training data
        # Preprocess the data (similar to what was done in splitdataset function)
        new_df.fillna(new_df.mean(), inplace=True)
        label_encoder = LabelEncoder()
        new_df['Date'] = label_encoder.fit_transform(new_df['Date'])
        X_new = new_df[['Date', 'MinTemp', 'MaxTemp', 'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']]
        
        print("New data processed successfully.")
        
        # Predict using the MLR model
        y_pred = mlr_model.predict(X_new)
        
        # Display the prediction results
        text.delete('1.0', END)
        text.insert(END, "Prediction Results\n\n")
        text.insert(END, f"Predicted results for the new data:\n{y_pred}\n\n")
        
        print("Prediction successful.")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction:\n{str(e)}")
        print("Error during prediction:", str(e))



# Prediction button with color orange
open_second_button = tk.Button(main,font=(13), text="Prediction", command=predict, bg="orange",width=16)
open_second_button.place(x=950, y=650)
open_second_button.config(font=font1)

import matplotlib.pyplot as plt



main.config(bg='#F08080')
main.mainloop()