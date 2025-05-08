import pandas as pd
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

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

def crawl_fixtures_fbref(league, season_str):
    """Crawl lịch thi đấu chưa diễn ra từ FBref."""
    print(f"Crawling fixtures for {league} season {season_str}...")
    league_ids = {
        "premierleague": (9, "Premier-League"),
        "laliga": (12, "La-Liga"),
        "seriea": (11, "Serie-A"),
        "bundesliga": (20, "Bundesliga"),
        "ligue1": (13, "Ligue-1"),
    }

    if league not in league_ids:
        print(f"League {league} không được hỗ trợ.")
        return []

    league_id, league_url = league_ids[league]
    url = f"https://fbref.com/en/comps/{league_id}/schedule/{season_str}-{league_url}-Scores-and-Fixtures"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Không thể truy cập {url} - status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", id=lambda x: x and x.startswith("sched_"))
    if not table:
        print("Không tìm thấy bảng lịch thi đấu.")
        return []

    fixtures = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 10:
            continue

        fixture = {
            "round": cells[0].text.strip(),             # Tuần
            "day": cells[1].text.strip(),               # Thứ
            "date": cells[2].text.strip(),              # Ngày cụ thể
            "time": cells[3].text.strip(),              # Giờ đá
            "home_team": cells[4].text.strip(),         # Đội nhà
            "away_team": cells[8].text.strip(),         # Đội khách
            "venue": cells[10].text.strip(),            # Địa điểm
            "league": league,
            "season": season_str
        }
        fixtures.append(fixture)

    return fixtures

# Sử dụng các hàm để crawl dữ liệu
# leagues = ["premierleague"]
# seasons = [2022] # Thay đổi thành các mùa giải khác nếu cần

leagues = ["premierleague", "laliga", "seriea", "bundesliga", "ligue1"]
seasons = list(range(2014, 2024))  # 2014–2023 -> mùa 2014–2015 đến 2023–2024

all_match_results = []
all_team_stats = []
all_fixtures = []

for league in leagues:
    for season in seasons:
        print(f"Đang crawl dữ liệu {league} mùa {season}...")
        
        # Crawl kết quả trận đấu
        match_results = crawl_match_results(league, season)
        all_match_results.extend(match_results)
        
        # Crawl thống kê đội bóng
        # team_stats = crawl_team_stats(league, season)
        # all_team_stats.extend(team_stats)

         # Crawl lịch thi đấu từ FBref
        season_str = f"{season}-{season+1}"
        fixtures = crawl_fixtures_fbref(league, season_str)
        all_fixtures.extend(fixtures)
        
        # Tránh bị chặn
        time.sleep(10)

# Lưu dữ liệu vào file CSV
pd.DataFrame(all_match_results).to_csv("match_results.csv", index=False)
# pd.DataFrame(all_team_stats).to_csv("team_stats.csv", index=False)
pd.DataFrame(all_fixtures).to_csv("data/fixtures.csv", index=False)
