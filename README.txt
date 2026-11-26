PREDICTOR LIGUE 1 BACKEND PATCH

Files:
- core.py: existing engines preserved; adds Ligue 1 historical loader, matching
  rolling-feature generator, team list, and prediction function.
- app.py: registers league code L1, loads models/ligue1_model.joblib,
  maps Football-data.org competition L1 -> FL1, and connects teams,
  manual predictions, model info, and fixtures endpoints.

Install:
1. Back up your current project files:
   app.py -> app.py.backup
   core.py -> core.py.backup
2. Copy these two files into:
   C:\xampp\htdocs\predictor\
3. Confirm these files exist:
   models\ligue1_model.joblib
   ligue1_training\data\raw\FR1_1011.csv ... FR1_2425.csv
4. Restart the FastAPI server.

Important:
- Do not run train_ligue1.py; the existing model is loaded, not retrained.
- Live fixtures still require FOOTBALL_DATA_API_TOKEN in the project-root .env.
- Football-data.org may deny access to Ligue 1 depending on the account/API plan.
- The existing EPL, LaLiga and Bundesliga model logic was not intentionally altered.
- Python syntax compilation passed. Ligue 1 rolling-feature function passed a
  small synthetic-data smoke test. Full local raw-data/API test must be run on
  the user's PC because their raw FR1 folder and project environment are not
  mounted in this workspace.
