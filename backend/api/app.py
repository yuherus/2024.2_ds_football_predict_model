from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sqlalchemy import text
import sys
import os

# Thêm đường dẫn chính của project vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các tiện ích cần thiết
from backend.features.utils import get_pg_engine

app = FastAPI(title="Football Prediction API", 
              description="API for football match prediction results",
              version="1.0.0")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins trong môi trường phát triển
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models Pydantic
class MatchPrediction(BaseModel):
    match_date: str
    round: Optional[int] = None
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_win_pct: str
    draw_pct: str
    away_win_pct: str
    league: Optional[str] = None
    season: Optional[str] = None
    venue: Optional[str] = None
    prediction_model: str
    predicted_result: int
    actual_result: Optional[int] = None

class PredictionResponse(BaseModel):
    total: int
    predictions: List[MatchPrediction]

@app.get("/")
async def root():
    return {"message": "Football Prediction API is running"}

@app.get("/predictions/", response_model=PredictionResponse)
async def get_predictions(
    league: Optional[str] = Query(None, description="Filter by league name"),
    team: Optional[str] = Query(None, description="Filter by team name (home or away)"),
    limit: int = Query(10, description="Number of predictions to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    upcoming_only: bool = Query(False, description="Only return upcoming matches")
):
    """
    Lấy danh sách các dự đoán trận đấu từ database.
    Có thể lọc theo giải đấu, đội bóng, và chỉ lấy các trận sắp tới.
    """
    try:
        engine = get_pg_engine()
        
        # Xây dựng câu truy vấn cơ bản
        query = """
        SELECT * FROM match_predictions
        WHERE 1=1
        """
        
        params = {}
        
        # Thêm các điều kiện lọc
        if league:
            query += " AND league = :league"
            params['league'] = league
            
        if team:
            query += " AND (home_team = :team OR away_team = :team)"
            params['team'] = team
            
        if upcoming_only:
            query += " AND match_date > CURRENT_TIMESTAMP"
        
        # Sắp xếp và phân trang
        query += " ORDER BY match_date DESC LIMIT :limit OFFSET :offset"
        params['limit'] = limit
        params['offset'] = offset
        
        # Đếm tổng số kết quả (không có giới hạn và offset)
        count_query = f"""
        SELECT COUNT(*) as total FROM match_predictions
        WHERE 1=1
        """
        
        if league:
            count_query += " AND league = :league"
        if team:
            count_query += " AND (home_team = :team OR away_team = :team)"
        if upcoming_only:
            count_query += " AND match_date > CURRENT_TIMESTAMP"
        
        # Thực hiện truy vấn
        with engine.connect() as conn:
            # Đếm tổng số
            count_result = conn.execute(text(count_query), params).fetchone()
            total = count_result[0] if count_result else 0
            
            # Lấy dữ liệu chi tiết
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            
            # Chuyển đổi kết quả thành danh sách dict
            predictions = []
            for row in rows:
                prediction = dict(row._mapping)
                # Đảm bảo đúng định dạng khi trả về JSON
                prediction['match_date'] = prediction['match_date'].isoformat() if prediction['match_date'] else None
                prediction['prediction_timestamp'] = prediction['prediction_timestamp'].isoformat() if prediction['prediction_timestamp'] else None
                predictions.append(prediction)
        
        return {"total": total, "predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 