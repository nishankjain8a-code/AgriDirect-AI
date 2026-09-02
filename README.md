# AgriDirect AI — SIH Live Demo

AI-assisted crop inspection and farmer decision support demo.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face Spaces

Create a new **Streamlit** Space and upload/push this repository.

The app is designed to run in Demo Mode without external APIs. Market/economic values are explicitly simulated for demonstration; the visual crop gate is a heuristic screening layer, not a production agronomic model.

## Important

For production deployment, replace the demo visual quality logic with an independently validated field-trained crop/disease/quality model and real market/weather/logistics APIs.
