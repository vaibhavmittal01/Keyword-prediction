import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import CENTER

data = None  
model = None  
vectorizer = None  

def load_model_and_vectorizer():
    try:
        model_info = joblib.load('best_model.pkl')

        global model, vectorizer  
        vectorizer = model_info['vectorizer']
        model = model_info['model']
        model_name = model_info['name']
        model_accuracy = model_info['accuracy']

        return model_name, model_accuracy
    except KeyError as e:
        messagebox.showerror("Error", f"KeyError: {str(e)}. Check the structure of best_model.pkl.")
        print(f"KeyError: {str(e)}. Check the structure of best_model.pkl.")
        return None, None
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model or vectorizer: {str(e)}")
        print(f"Error loading model or vectorizer: {str(e)}")
        return None, None

def predict_keywords_from_csv(input_csv, tree):
    global model, vectorizer, data  
    try:
        new_data = pd.read_csv(input_csv)
        if 'Text' not in new_data.columns:
            raise ValueError("Input CSV must contain a 'Text' column.")
        texts_tfidf = vectorizer.transform(new_data['Text'])
        predictions = model.predict(texts_tfidf)
        probabilities = model.predict_proba(texts_tfidf)
        new_data['Predicted Label'] = predictions
        new_data['Accuracy'] = [max(probs) for probs in probabilities]  

        data = new_data

        
        tree.delete(*tree.get_children())

        
        for idx, row in new_data.iterrows():
            tree.insert("", "end", values=(row['Text'], row['Predicted Label'], f"{row['Accuracy']:.2f}"))

        return new_data
    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"Error in prediction: {str(e)}")
        return None

def open_file_dialog_train(tree):
    input_csv = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if input_csv:
        model_name, model_accuracy = train_model(input_csv)
        if model_name and model_accuracy:
            messagebox.showinfo("Success", f"Model trained successfully using {model_name} with accuracy {model_accuracy:.2f}%")

def open_file_dialog_predict(tree):
    input_csv = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if input_csv:
        model_name, model_accuracy = load_model_and_vectorizer()
        if model:
            data = predict_keywords_from_csv(input_csv, tree)
            if data is not None:
                result_label.config(text=f"Model: {model_name}\nAccuracy: {model_accuracy:.2f}%")
                messagebox.showinfo("Success", "Predictions generated successfully.")

def download_csv(model, vectorizer):  
    global data  
    if data is not None:
        output_csv = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if output_csv:
            
            data.to_csv(output_csv, index=False)
            messagebox.showinfo("Success", f"Data saved to {output_csv}")

def download_train_dummy_csv():
    dummy_data = pd.DataFrame({"Text": ["Sample text 1","Sample Text 2","Sample Text 3"], 
                            "label":["Sample label 1","Sample label 2","Sample label 3"]})
    dummy_csv_path="dummy_train_data.csv"
    dummy_data.to_csv(dummy_csv_path, index=False)
    messagebox.showinfo("Dummy Data Created",f"Dummy data saved to {dummy_csv_path}")
    os.system("start "+ dummy_csv_path)
    
    
def download_dummy_csv():
    
    dummy_data = pd.DataFrame({"Text": ["Sample text 1", "Sample text 2", "Sample text 3"]})
    dummy_csv_path = "dummy_data.csv"
    dummy_data.to_csv(dummy_csv_path, index=False)
    messagebox.showinfo("Dummy Data Created", f"Dummy data saved to {dummy_csv_path}")

    
    os.system("start " + dummy_csv_path)

def train_model(input_csv):
    global model, vectorizer
    try:
        
        train_data = pd.read_csv(input_csv)
        X_train, X_test, y_train, y_test = train_test_split(train_data['Text'], train_data['label'], test_size=0.2, random_state=42)

        
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_tfidf, y_train)
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_tfidf))

        
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train_tfidf, y_train)
        gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test_tfidf))

        
        if rf_accuracy > gb_accuracy:
            model = rf_model
            model_name = 'Random Forest'
            model_accuracy = rf_accuracy*100
        else:
            model = gb_model
            model_name = 'Gradient Boosting'
            model_accuracy = gb_accuracy*100

        
        joblib.dump({'model': model, 'vectorizer': vectorizer, 'name': model_name, 'accuracy': model_accuracy}, 'best_model.pkl')

        return model_name, model_accuracy
    except Exception as e:
        messagebox.showerror("Error", f"Error training model: {str(e)}")
        print(f"Error training model: {str(e)}")
        return None, None

root = tk.Tk()
root.title("Prediction Using Bucketing")
root.geometry("800x600")

style = ttk.Style()

style.configure("Background.TFrame", background="#87CEFA")

style.configure("TButton", font=("Helvetica", 12), padding=10, background="#87CEFA", foreground="black")  
style.configure("TLabel", font=("Helvetica", 16), background="#87CEFA", foreground="black")  
style.configure("TFrame", padding=20, background="#f0f0f0")  

frame = ttk.Frame(root, style="Background.TFrame")
frame.pack(fill="both", expand=True)
frame.pack_propagate(False)  

logo_image= tk.PhotoImage(file="Techmagnate_logo2.png")
logo_label= ttk.Label(frame, image= logo_image)
logo_label.pack(anchor="nw", padx=10, pady=10)

label = ttk.Label(frame, text="Select a CSV file to train the model containing Text and Label(target data) columns:", style="TLabel")
label.pack(pady=10)

button_frame1= ttk.Frame(frame)
button_frame1.pack(pady=10)

btn_train = ttk.Button(button_frame1, text="Upload Training Data", command=lambda: open_file_dialog_train(tree), style="TButton")
btn_train.pack(side="left")

btn_dummy_download = ttk.Button(button_frame1, text="Download Train Data Dummy CSV File(For Format Reference)", command=download_train_dummy_csv, style="TButton")
btn_dummy_download.pack(side="left")

label = ttk.Label(frame, text="Upload test data CSV file for predictions with column containing Text:", style="TLabel")
label.pack(pady=10)

button_frame = ttk.Frame(frame)
button_frame.pack(pady=10)

btn_load = ttk.Button(button_frame, text="Upload test data CSV file", command=lambda: open_file_dialog_predict(tree), style="TButton")
btn_load.pack(side="left")

btn_dummy_download = ttk.Button(button_frame, text="Download Test Data Dummy CSV File(For Format Reference)", command=download_dummy_csv, style="TButton")
btn_dummy_download.pack(side="left")

btn_exit = ttk.Button(frame, text="Exit", command=root.quit)
btn_exit.pack(pady=10)

result_label = ttk.Label(frame, text="Resulted Prediction", font=("Helvetica", 12), style="TLabel")
result_label.pack(pady=10)


tree = ttk.Treeview(frame, columns=("Text", "Predicted Label", "Accuracy"), show="headings", height=10)
tree.heading("Text", text="Text", anchor=tk.CENTER)
tree.heading("Predicted Label", text="Predicted Label", anchor=tk.CENTER)
tree.heading("Accuracy", text="Accuracy", anchor=tk.CENTER)  


tree.column("Text", anchor=tk.CENTER)
tree.column("Predicted Label", anchor=tk.CENTER)
tree.column("Accuracy", anchor=tk.CENTER)


scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
scrollbar.pack(side="right", fill="y")
tree.configure(yscrollcommand=scrollbar.set)

tree.pack(pady=10)


btn_download = ttk.Button(frame, text="Download the Predicted Data in CSV format", command=lambda: download_csv(model, vectorizer), style="TButton")
btn_download.pack(pady=10)

root.mainloop()
