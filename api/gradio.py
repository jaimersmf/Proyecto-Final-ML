import gradio as gr
import requests

def predict_gradio(product, state, company, days_to_company, timely_response):
    data = {
        "Product": product,
        "State": state,
        "Company": company,
        "days_to_company": days_to_company,
        "Timely_response": timely_response
    }
    r = requests.post("http://127.0.0.1:8000/predict", json=data)
    return r.json()

iface = gr.Interface(
    fn=predict_gradio,
    inputs=["text", "text", "text", "number", "number"],
    outputs="json"
)

iface.launch()
