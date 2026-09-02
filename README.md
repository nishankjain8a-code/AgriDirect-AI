# AgriDirect AI — SIH Streamlit Simulator

A browser-first SIH demo for the AgriDirect AI concept. It supports crop image upload/camera capture and a deterministic simulation of crop intelligence, quality, market, sell-vs-hold, buyer/FPO and decision-receipt flows.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload this folder to a GitHub repository and deploy `app.py` using Streamlit Community Cloud. The entry point is `app.py`.

## Important

The simulator deliberately labels market/economic values as simulated. For production, connect validated Indian field-trained vision models plus live market, weather, logistics and buyer/FPO data sources.
