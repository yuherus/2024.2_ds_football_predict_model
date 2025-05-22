# Football Prediction API

API để hiển thị kết quả dự đoán trận đấu bóng đá trên website.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Khởi chạy API

```bash
cd backend/api
python run.py
```

Server sẽ chạy tại địa chỉ: http://localhost:8000

## Tài liệu API

Sau khi khởi chạy server, bạn có thể truy cập tài liệu Swagger UI tại:

- http://localhost:8000/docs

Hoặc tài liệu ReDoc:

- http://localhost:8000/redoc

## Các endpoint

### 1. Lấy danh sách dự đoán

```
GET /predictions/
```

Tham số:
- `league`: (tùy chọn) Lọc theo giải đấu 
- `team`: (tùy chọn) Lọc theo đội bóng (home hoặc away)
- `limit`: Số lượng kết quả trả về (mặc định: 10)
- `offset`: Vị trí bắt đầu cho phân trang (mặc định: 0)
- `upcoming_only`: Chỉ lấy các trận sắp diễn ra (mặc định: false)

Ví dụ:
```
GET /predictions/?league=premierleague&team=Arsenal&limit=20&upcoming_only=true
```

### 2. Lấy danh sách trận đấu sắp tới

```
GET /predictions/upcoming/
```

Tham số:
- `days`: Số ngày tới muốn xem lịch (mặc định: 7)
- `league`: (tùy chọn) Lọc theo giải đấu
- `team`: (tùy chọn) Lọc theo đội bóng

Ví dụ:
```
GET /predictions/upcoming/?days=14&league=laliga
```

## Ví dụ phản hồi

```json
{
  "total": 10,
  "predictions": [
    {
      "id": 1,
      "match_date": "2024-03-20T15:00:00Z",
      "round": 29,
      "home_team": "Manchester City",
      "away_team": "Arsenal",
      "home_win_prob": 0.523,
      "draw_prob": 0.231,
      "away_win_prob": 0.246,
      "home_win_pct": "52.3%",
      "draw_pct": "23.1%",
      "away_win_pct": "24.6%",
      "league": "premierleague",
      "season": "2023-2024",
      "venue": "Etihad Stadium",
      "prediction_model": "xgboost",
      "predicted_result": 2,
      "actual_result": null,
      "prediction_timestamp": "2024-03-15T10:30:45Z"
    },
    ...
  ]
}
```

## Chú thích giá trị predicted_result và actual_result

- 0: Đội khách thắng
- 1: Hòa
- 2: Đội nhà thắng 