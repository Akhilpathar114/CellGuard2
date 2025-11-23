
# CellGuard.AI - Deployable Streamlit App

Files:
- app.py - Streamlit application (entrypoint)
- requirements.txt - Python packages required
- sample_data.csv - example dataset (optional) 

## Deploy on Streamlit Cloud
1. Put `app.py` and `requirements.txt` at the repository root.
2. Commit and push to your git repo.
3. On Streamlit Cloud, create a new app and point to the repository and branch.
4. Ensure the "Main file" is `app.py`.
5. Deploy. Watch the build logs to confirm `plotly` is installed.

## Run locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
