import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """Thiết lập WebDriver với các tùy chọn phù hợp."""
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def crawl_match_results(league, season):
    """Crawl kết quả trận đấu cho một giải đấu và mùa giải cụ thể."""
    print(f"Crawling match results for {league} season {season}...")
    driver = setup_driver()
    results = []
    
    try:
        # Ví dụ: Crawl kết quả từ Transfermarkt
        url = f"https://www.transfermarkt.com/premier-league/gesamtspielplan/wettbewerb/GB1?saison_id={season}"
        if league == "bundesliga":
            url = f"https://www.transfermarkt.com/bundesliga/gesamtspielplan/wettbewerb/L1?saison_id={season}"
        elif league == "laliga":
            url = f"https://www.transfermarkt.com/laliga/gesamtspielplan/wettbewerb/ES1?saison_id={season}"
        elif league == "seriea":
            url = f"https://www.transfermarkt.com/serie-a/gesamtspielplan/wettbewerb/IT1?saison_id={season}"
        elif league == "ligue1":
            url = f"https://www.transfermarkt.com/ligue-1/gesamtspielplan/wettbewerb/FR1?saison_id={season}"
        
        print(f"Đang truy cập {url}...")
        driver.get(url)
        time.sleep(10)
        
        # In pagesource ra 1 file
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        # Lấy tất cả các trận đấu thẻ tr mà không có class bg_blau_20
        match_elements = driver.find_elements(By.CSS_SELECTOR, "table tbody tr:not(.bg_blau_20)")
        for match in match_elements:
            try:
                # Tìm ngày tháng
                
                date_elements = match.find_elements(By.CSS_SELECTOR, "td.hide-for-small a")
                date = date_elements[0].text if date_elements else "N/A"
                if date != "N/A":
                    print(f"Ngày: {date}")
                
                # Tìm đội nhà
                home_team_element = match.find_element(By.CSS_SELECTOR, "td.text-right.no-border-rechts.hauptlink a")
                home_team = home_team_element.text
                
                # Tìm đội khách
                away_team_element = match.find_element(By.CSS_SELECTOR, "td.no-border-links.hauptlink a")
                away_team = away_team_element.text
                
                # Tìm tỷ số
                score_element = match.find_element(By.CSS_SELECTOR, "td.zentriert.hauptlink a.ergebnis-link")
                score = score_element.text
                
                # Kiểm tra nếu trận đấu đã diễn ra
                if score and ":" in score:
                    home_score, away_score = score.split(":")
                    
                    match_data = {
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": int(home_score),
                        "away_score": int(away_score),
                        "league": league,
                        "season": season
                    }
                    
                    results.append(match_data)
                    print(f"Đã xử lý: {home_team} {score} {away_team}")
            except Exception as e:
                print(f"Lỗi khi xử lý trận đấu: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Lỗi khi crawl {league} mùa {season}: {str(e)}")
    finally:
        driver.quit()
        
    return results

def crawl_team_stats(league, season):
    """Crawl thống kê đội bóng."""
    # Tương tự như hàm trên

def crawl_player_stats(league, season):
    """Crawl thông tin cầu thủ và phong độ."""
    # Tương tự như các hàm trên
    # ...

# Sử dụng các hàm để crawl dữ liệu
leagues = ["premierleague"]
seasons = [2022] # Thay đổi thành các mùa giải khác nếu cần

all_match_results = []
all_team_stats = []

for league in leagues:
    for season in seasons:
        print(f"Đang crawl dữ liệu {league} mùa {season}...")
        
        # Crawl kết quả trận đấu
        match_results = crawl_match_results(league, season)
        all_match_results.extend(match_results)
        
        # Crawl thống kê đội bóng
        # team_stats = crawl_team_stats(league, season)
        # all_team_stats.extend(team_stats)
        
        # Tránh bị chặn
        time.sleep(10)

# Lưu dữ liệu vào file CSV
pd.DataFrame(all_match_results).to_csv("match_results.csv", index=False)
# pd.DataFrame(all_team_stats).to_csv("team_stats.csv", index=False)
