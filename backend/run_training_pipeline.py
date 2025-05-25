import os
import time
import sys
import importlib
from datetime import datetime

def run_module(module_name):
    """Import and run a Python module by name"""
    try:
        # Import module by name
        module = importlib.import_module(module_name)
        
        # If the module has a main function, call it
        if hasattr(module, 'main'):
            module.main()
        # If the module has a train_all_models function, call it
        elif hasattr(module, 'train_all_models'):
            module.train_all_models()
        else:
            print(f"Warning: No main() or train_all_models() function found in {module_name}")
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        return False
    except Exception as e:
        print(f"Error running module {module_name}: {e}")
        return False
    
    return True

def log_step(message):
    """Log a step in the pipeline with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] {message}")
    print("=" * 80)

def main():
    """Run the complete training pipeline"""
    start_time = time.time()
    
    log_step("Starting Football Prediction Model Training Pipeline")
    
    # Step 1: Add squad value features
    log_step("Step 1: Adding squad value features")
    if not run_module("backend.features.add_squad_value_features"):
        print("Warning: Failed to run squad value feature module. Skipping this step.")
    
    # Bước mới: Tiền xử lý nâng cao
    log_step("Step 2: Running enhanced preprocessing")
    print("\nEnhanced Preprocessing:")
    print("- Điền các round còn thiếu dựa trên nội suy từ ngày tháng")
    print("- Điền các thông số trận đấu bằng trung vị của đội bóng trong mùa")
    print("- Xử lý outliers bằng phương pháp IQR")
    print("- Chuẩn hóa dữ liệu với RobustScaler")
    
    if not run_module("backend.features.enhanced_preprocessing"):
        print("Warning: Failed to run enhanced preprocessing module. Skipping this step.")
    
    # Step 3: Train models (was Step 2 before)
    log_step("Step 3: Training prediction models with enhanced architecture")
    print("\nXGBoost Model:")
    print("- Sử dụng GridSearchCV để tìm tham số tối ưu")
    print("- Tối ưu các tham số: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree")
    print("- Tự động điều chỉnh scale_pos_weight dựa trên phân phối lớp")
    print("- Tạo biểu đồ phân tích ảnh hưởng của các tham số")
    print("- Lưu tham số tốt nhất vào file JSON để tái sử dụng")
    print("- Tự động tải tham số đã lưu để tiết kiệm thời gian")
    
    print("\nLSTM Model:")
    print("- Bidirectional LSTM layers")
    print("- Attention mechanism")
    print("- Batch normalization")
    print("- Dropout optimization")
    print("- Class weight balancing")
    print("- Learning rate scheduling")
    print("- Enhanced feature extraction")
    
    # Thêm tham số dòng lệnh để bắt buộc tìm kiếm lại tham số
    force_grid_search = "--force-grid-search" in sys.argv
    if force_grid_search:
        print("\nChú ý: Đã kích hoạt tìm kiếm lại tham số (--force-grid-search)")
    
    if not run_module("backend.models.model_training"):
        print("Error: Failed to run model training module. Pipeline failed.")
        return
    
    # Calculate and display total runtime
    total_runtime = time.time() - start_time
    hours, remainder = divmod(total_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    log_step(f"Pipeline completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nResults can be found in the 'models' directory.")
    print("Preprocessed data can be found in 'backend/data/preprocessed_matches.csv'")
    print("XGBoost Grid Search results can be found in 'models/grid_search_results'.")
    print("XGBoost Best Parameters can be found in 'models/grid_search_results/best_params'.")
    print("\nĐể chạy lại tìm kiếm tham số, sử dụng tham số --force-grid-search:")
    print("python -m backend.run_training_pipeline --force-grid-search")

if __name__ == "__main__":
    main() 