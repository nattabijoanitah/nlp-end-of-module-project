import gradio as gr
from transformers import pipeline

# In Colab/production, point to your Hugging Face model or local saved directory
MODEL_ID = "sentiment_analyzer/model"
try:
    clf = pipeline("text-classification", model=MODEL_ID)
except Exception:
    clf = None

def predict(text):
    if not text.strip():
        return {}
    if clf:
        r = clf(text)[0]
        other = "NEGATIVE" if r["label"] == "POSITIVE" else "POSITIVE"
        return {r["label"]: round(float(r["score"]), 4), other: round(1.0 - float(r["score"]), 4)}
    else:
        # Fallback demonstration
        pos = 0.95 if "good" in text.lower() or "amazing" in text.lower() else 0.10
        return {"POSITIVE": pos, "NEGATIVE": 1.0 - pos}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence..."),
    outputs=gr.Label(num_top_classes=2),
    title="IMDb Sentiment Analyser",
    examples=["The food was amazing!", "Worst service ever."]
)

if __name__ == "__main__":
    demo.launch()
