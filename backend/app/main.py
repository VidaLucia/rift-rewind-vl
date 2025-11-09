from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   

from backend.app.routers import (
    health_router,
    predict_router,
    sheet_router,
    progression_router,
    levelup_router,
    player_router,
    timeline_pipeline_router,
    bedrock_summary_router,
)

app = FastAPI(
    title="League → D&D Backend",
    version="1.0.0",
    description="Backend for transforming Riot match data into D&D sheets and summaries.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; replace with frontend URL when deploying to AWS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router.router, prefix="/health", tags=["Health"])
app.include_router(predict_router.router, prefix="/predict", tags=["Prediction"])
app.include_router(sheet_router.router, prefix="/sheet", tags=["Character Sheets"])
app.include_router(progression_router.router, prefix="/progression", tags=["Progression"])
app.include_router(levelup_router.router, prefix="/levelup", tags=["Level Up"])
app.include_router(player_router.router, prefix="/player", tags=["Player"])
app.include_router(timeline_pipeline_router.router, prefix="/timeline", tags=["Timeline"])
app.include_router(bedrock_summary_router.router, prefix="/summary", tags=["Bedrock Summary"])

