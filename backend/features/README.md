# Football Prediction System - Data Processing Module

Mô-đun xử lý dữ liệu của hệ thống dự đoán kết quả bóng đá. Module này thực hiện việc tải, xử lý và lưu trữ dữ liệu bóng đá vào cơ sở dữ liệu PostgreSQL.

## Cấu trúc thư mục

```
backend/features/
│
├── add_squad_value_features.py - Thêm giá trị đội hình vào dữ liệu
├── config.py                   - Cấu hình cho các module 
├── data_loader.py              - Module tải dữ liệu từ CSV vào DB
├── db_setup.py                 - Thiết lập cấu trúc cơ sở dữ liệu
├── feature_engineering.py      - Tạo đặc trưng cho mô hình dự đoán
├── main_processor.py           - Module xử lý chính điều phối quá trình
└── utils.py                    - Các tiện ích dùng chung
```

## Cài đặt

### Yêu cầu

- Python 3.8 trở lên
- PostgreSQL 12 trở lên
- Các thư viện Python: pandas, numpy, sqlalchemy, psycopg2-binary

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

### Cấu hình cơ sở dữ liệu

1. Mở file `config.py` và cập nhật thông tin kết nối PostgreSQL:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'dbname': 'football_prediction',
    'user': 'username',
    'password': 'password'
}
```

2. Cập nhật đường dẫn tới file dữ liệu CSV (vẫn trong `config.py`):

```python
# Đường dẫn tới dữ liệu đội bóng
ALL_PLAYERS_CSV_PATH = 'path/to/all_players.csv'

# Đường dẫn tới dữ liệu trận đấu
MATCH_RESULTS_CSV_PATH = 'path/to/match_results.csv'
```

## Sử dụng

### Chạy toàn bộ pipeline

Để chạy toàn bộ quy trình xử lý dữ liệu:

```bash
python -m backend.features.main_processor
python -m backend.features.add_squad_value_features
```

