# Mô hình Dự đoán Kết quả Bóng đá

Dự án này sử dụng hai mô hình học máy (XGBoost và LSTM) để dự đoán kết quả các trận đấu bóng đá dựa trên dữ liệu lịch sử.

## Yêu cầu hệ thống

- Python 3.8+
- PostgreSQL
- Các thư viện Python cần thiết (xem file `requirements.txt`)

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Đảm bảo PostgreSQL đã được cài đặt và cấu hình đúng trong `backend/features/config.py`

## Luồng xử lý dữ liệu

Dự án được tổ chức theo các bước sau:

1. **Xử lý dữ liệu** (trong folder `features`):
   - Tải và xử lý dữ liệu từ file CSV
   - Lưu vào PostgreSQL
   - Tạo các đặc trưng cần thiết cho việc huấn luyện

2. **Huấn luyện mô hình** (trong folder `models`):
   - Chuẩn bị dữ liệu từ PostgreSQL
   - Huấn luyện mô hình XGBoost và LSTM
   - Đánh giá và so sánh hiệu suất của các mô hình
   - Lưu mô hình tốt nhất

## Phương pháp chia dữ liệu

Dự án sử dụng phương pháp chia dữ liệu theo thời gian (time-based split) cho từng giải đấu riêng biệt:

- **Tập huấn luyện (train)**: 7 mùa đầu tiên của mỗi giải đấu
- **Tập xác thực (validation)**: 2 mùa tiếp theo của mỗi giải đấu
- **Tập kiểm tra (test)**: 1 mùa cuối cùng của mỗi giải đấu

Cách chia này giúp mô phỏng tốt hơn tình huống dự đoán thực tế, khi chúng ta sử dụng dữ liệu lịch sử để dự đoán kết quả tương lai, đồng thời bảo toàn đặc thù của từng giải đấu.

## Đặc trưng mới

Dự án đã bổ sung hai đặc trưng mới về giá trị thị trường của đội hình ra sân:

- **home_squad_value**: Tổng giá trị thị trường của cầu thủ trong đội hình đội nhà
- **away_squad_value**: Tổng giá trị thị trường của cầu thủ trong đội hình đội khách

Các đặc trưng này được tính từ cột `home_lineup` và `away_lineup` bằng cách:
1. Xác định các cầu thủ trong đội hình xuất phát
2. Tìm giá trị thị trường của từng cầu thủ theo đúng mùa giải tương ứng
3. Tính tổng giá trị của đội hình

## Cấu trúc folder `models`

- `config.py`: Cấu hình cho các mô hình
- `data_preparation.py`: Chuẩn bị dữ liệu từ PostgreSQL và chia dữ liệu theo thời gian
- `xgboost_model.py`: Mô hình XGBoost
- `lstm_model.py`: Mô hình LSTM
- `model_evaluation.py`: Đánh giá hiệu suất
- `model_storage.py`: Lưu và tải mô hình
- `model_training.py`: Huấn luyện mô hình theo phương pháp time-based
- `compare_models.py`: So sánh các mô hình

## Hướng dẫn sử dụng

### 1. Cấu hình PostgreSQL

Trước khi bắt đầu, hãy đảm bảo PostgreSQL đã được cài đặt và cấu hình trong file `backend/features/config.py`:

```python
DB_CONFIG = {
    "dbname": "your_database_name",
    "user": "your_username",
    "password": "your_password",
    "host": "localhost",
    "port": 5432
}
```

### 2. Chuẩn bị dữ liệu

Đầu tiên, cần xử lý dữ liệu từ CSV và lưu vào PostgreSQL. Chạy:

```bash
cd backend
python -m features.main_processor
```

Quá trình này sẽ tạo các bảng trong PostgreSQL và lưu dữ liệu đã xử lý.

### 3. Tạo đặc trưng giá trị đội hình

Để tạo đặc trưng mới về giá trị thị trường của đội hình, chạy:

```bash
cd backend
python -m features.add_squad_value_features
```

Script này sẽ tính toán giá trị đội hình cho mỗi trận đấu và cập nhật vào cơ sở dữ liệu.

### 4. Huấn luyện mô hình

Để huấn luyện và so sánh các mô hình, chạy:

```bash
cd backend
python -m models.model_training
```

Script này sẽ:

- Tải dữ liệu từ PostgreSQL
- Chia dữ liệu theo phương pháp time-based cho từng giải đấu
- Chuẩn bị dữ liệu cho cả hai mô hình
- Huấn luyện mô hình XGBoost và LSTM riêng cho từng giải đấu
- Đánh giá hiệu suất của từng mô hình
- Tạo các biểu đồ so sánh
- Lưu các mô hình đã huấn luyện

### 5. Xem kết quả đánh giá

Kết quả đánh giá sẽ được hiển thị trên terminal và các biểu đồ so sánh sẽ được lưu trong thư mục `models/<league_name>` và `models/plots`.

## Điều chỉnh tham số

Bạn có thể điều chỉnh các tham số của mô hình trong file `config.py`:

- `XGBOOST_PARAMS`: Tham số của mô hình XGBoost
- `LSTM_SEQUENCE_LENGTH`: Độ dài chuỗi cho LSTM
- `LSTM_BATCH_SIZE`: Kích thước batch cho LSTM
- `LSTM_EPOCHS`: Số epochs cho LSTM

## Lưu ý quan trọng

1. Đảm bảo PostgreSQL đang chạy khi thực hiện các script.
2. Dữ liệu trong bảng `matches_featured` phải được tạo trước khi huấn luyện mô hình.
3. Quá trình huấn luyện có thể mất nhiều thời gian tùy thuộc vào lượng dữ liệu và cấu hình máy tính.
4. Cần có ít nhất 10 mùa giải cho mỗi giải đấu để phương pháp chia dữ liệu time-based hoạt động tốt. Nếu không đủ 10 mùa, giải đấu đó có thể bị bỏ qua.
5. Việc huấn luyện mô hình riêng cho từng giải đấu giúp mô hình nắm bắt tốt hơn đặc thù của giải đấu đó (ví dụ: phong cách thi đấu, mức độ cạnh tranh).
6. Đặc trưng giá trị đội hình có thể bị thiếu cho một số trận đấu không có thông tin đội hình đầy đủ.

## Yêu cầu phần cứng

- RAM: ít nhất 8GB
- CPU: 4 cores trở lên (để xử lý nhanh hơn)
- GPU: Không bắt buộc nhưng có thể tăng tốc độ huấn luyện LSTM

## Vấn đề thường gặp

**Lỗi kết nối PostgreSQL**: Đảm bảo PostgreSQL đang chạy và cấu hình kết nối đúng.

**Lỗi thiếu thư viện**: Chạy `pip install -r requirements.txt` để cài đặt các thư viện cần thiết.

**Lỗi thiếu dữ liệu**: Đảm bảo đã chạy `features.main_processor` để chuẩn bị dữ liệu.

**Lỗi không đủ mùa giải**: Đảm bảo dữ liệu có đủ 10 mùa giải cho mỗi giải đấu. Các giải đấu không đủ mùa sẽ bị bỏ qua trong quá trình huấn luyện.
