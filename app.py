# Loan Approval Prediction App (Gradio)
import pandas as pd
import pickle
import gradio as gr

# LOAD TRAINED MODEL

MODEL_PATH = "loan_approval_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("✅ Model loaded successfully")

# PREDICTION FUNCTION


def predict_loan(
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area
):
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }])

    prediction = model.predict(input_data)[0]

    return "✅ Loan Approved" if prediction == "Y" else "❌ Loan Rejected"


interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Married"),
        gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
        gr.Dropdown(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Coapplicant Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Amount Term"),
        gr.Number(label="Credit History (1 = Good, 0 = Bad)"),
        gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏦 Loan Approval Prediction System",
    description="Enter applicant details to predict loan approval status."
)


if __name__ == "__main__":
    interface.launch()
