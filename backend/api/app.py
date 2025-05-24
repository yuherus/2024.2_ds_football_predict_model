from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sqlalchemy import text
import sys
import os
from collections import defaultdict
import re

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

# def slugify(text: str) -> str:
#     """Chuyển chuỗi về dạng slug-friendly: 'Premier League' → 'premier-league'"""
#     text = text.lower()
#     text = re.sub(r"[^\w\s-]", "", text)  # loại bỏ ký tự đặc biệt
#     text = re.sub(r"[\s]+", "-", text)    # thay space bằng dấu gạch ngang
#     return text.strip("-")

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

@app.get("/matches/predictions/")
async def get_predictions(
    team: Optional[str] = Query(None, description="Filter by team name (home or away)"),
    limit: Optional[int] = Query(None, description="Number of predictions to return", ge=1, le=1000),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    upcoming_only: bool = Query(False, description="Only return upcoming matches")
):
    try:
        engine = get_pg_engine()

        # Truy vấn dữ liệu
        query = """
        SELECT * FROM match_predictions
        WHERE 1=1
        """
        params = {}

        if team:
            query += " AND (home_team = :team OR away_team = :team)"
            params["team"] = team
        if upcoming_only:
            query += " AND match_date > CURRENT_TIMESTAMP"

        query += " ORDER BY league, round, match_date DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        leagues_dict = {}

        # Gom nhóm theo league → round
        league_group = defaultdict(lambda: {
            "league_name": "",
            "total_matches": 0,
            "matches_by_round": defaultdict(list)
        })

        for row in rows:
            row_data = dict(row._mapping)
            league_name = row_data.get("league", "Unknown League")
            round_number = str(row_data.get("round", "unknown"))

            match = {
                "round": row_data.get("round"),
                "match_date": row_data.get("match_date").isoformat() if row_data.get("match_date") else None,
                "home_team": row_data.get("home_team"),
                "away_team": row_data.get("away_team"),
                "venue": row_data.get("venue"),
                "predictions": {
                    "home_win": row_data.get("home_win_prob"),
                    "draw": row_data.get("draw_prob"),
                    "away_win": row_data.get("away_win_prob")
                }
            }

            league_group[league_name]["league_name"] = league_name
            league_group[league_name]["matches_by_round"][round_number].append(match)
            league_group[league_name]["total_matches"] += 1

        # Đưa vào kết quả trả về
        for slug, league_data in league_group.items():
            leagues_dict[slug] = {
                "league_name": league_data["league_name"],
                "total_matches": league_data["total_matches"],
                "matches_by_round": league_data["matches_by_round"]
            }

        return {
            "success": True,
            "data": leagues_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@app.get("/championship_probabilities/")
async def get_championship_probabilities(
    limit: Optional[int] = Query(100, description="Số bản ghi trả về", ge=1, le=1000),
    offset: int = Query(0, description="Phân trang offset", ge=0),
):
    try:
        engine = get_pg_engine()

        query = """
        SELECT * FROM championship_probabilities
        ORDER BY league, rank, championship_probability DESC
        LIMIT :limit OFFSET :offset
        """
        params = {"limit": limit, "offset": offset}

        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()

        leagues_dict = {}

        for row in rows:
            row_data = dict(row._mapping)
            league_name_raw = row_data.get("league", "Unknown League")
            league_key = league_name_raw.lower().replace(" ", "")  # Ví dụ: 'Premier League' -> 'premierleague'

            # Cấu trúc như yêu cầu
            if league_key not in leagues_dict:
                leagues_dict[league_key] = {
                    "league_name": league_name_raw,
                    "teams": []
                }
            leagues_dict[league_key]["teams"].append({
                "rank": row_data.get("rank"),
                "team_name": row_data.get("team_name"),
                "championship_probability": row_data.get("championship_probability"),
                "points": row_data.get("points"),
                "form": row_data.get("form", "")  # nếu có cột form
            })

        return leagues_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 