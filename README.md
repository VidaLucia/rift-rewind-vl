# Rift Rewind – League of Legends Coaching Agent

A machine learning–driven coaching agent that analyzes **League of Legends** match data, predicts player archetypes, and generates dynamic **D&D-style character sheets** with personalized AI feedback.

##  Features
-  **HDBSCAN Role-Based Clustering** — groups players by playstyle and performance.  
-  **D&D Archetype Mapping** — assigns fantasy-inspired classes to each player.  
-  **AWS Bedrock LLM Integration** — produces narrative coaching feedback from match timelines.  
-  **Dynamic Character Sheets** — tracks player stats, spells, items, and level progression.  
-  **FastAPI + React Stack** — interactive web dashboard with cloud-ready backend.  

##  Data & Tools
- **Data Sources:** Riot Games Match V5 + Timeline V5 APIs  
- **AWS Services:** Bedrock (LLM), SageMaker (training and inference)  
- **Core Libraries:** Python, Pandas, HDBSCAN, UMAP, FastAPI, React  
## API Key
- Add your api KEY into the .env
##  Run Locally
```bash
# Backend
uvicorn backend.app.main:app --reload

# Frontend
cd rift-rewind-frontend
npm run dev
```
