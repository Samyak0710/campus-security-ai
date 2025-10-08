# Campus Security AI — Railway deployment package

This package contains a Streamlit MVP app ready to deploy on Railway.

## Files
- `app.py` - Streamlit app
- `data/` - sample data (CSV + JSON)
- `requirements.txt`
- `Dockerfile`
- `Procfile`
- `.railway.json` (example env settings)

## Quick Railway deploy steps
1. Create a new GitHub repo and push this project OR upload the folder to Railway directly.
2. On Railway dashboard -> New Project -> Deploy from GitHub (select repo) OR deploy using "Deploy from a GitHub repo".
3. Railway should detect the `Procfile`. Ensure the environment variable `PORT` is set to `8080` in the Railway project settings (or .railway.json will set it).
4. Deploy. The app will be available at `https://<your-railway-subdomain>.up.railway.app`

## Local run
1. python -m venv venv && source venv/bin/activate
2. pip install -r requirements.txt
3. streamlit run app.py --server.port 8080 --server.address 0.0.0.0
