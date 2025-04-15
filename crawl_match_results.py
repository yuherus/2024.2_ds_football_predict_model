import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")
    return webdriver.Chrome(options=chrome_options)

def crawl_fbref_match_results(league_name, league_id, league_slug, season):
    print(f"\n Crawling {league_name} - {season}-{season+1}...")
    driver = setup_driver()
    results = []

    try:
        url = f"https://fbref.com/en/comps/{league_id}/{season}-{season+1}/schedule/{season}-{season+1}-{league_slug}-Scores-and-Fixtures"
        driver.get(url)
        time.sleep(5)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.stats_table tbody tr")
        for row in rows:
            try:
                if row.get_attribute("class") == "thead":
                    continue

                date = row.find_element(By.CSS_SELECTOR, "td[data-stat='date']").text
                home_team = row.find_element(By.CSS_SELECTOR, "td[data-stat='home_team']").text
                away_team = row.find_element(By.CSS_SELECTOR, "td[data-stat='away_team']").text
                score = row.find_element(By.CSS_SELECTOR, "td[data-stat='score']").text

                if score and "–" in score:
                    home_score, away_score = map(int, score.split("–"))
                    match_data = {
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "league": league_name.lower().replace(" ", ""),
                        "season": f"{season}-{season+1}",
                        "source": "FBref"
                    }
                    results.append(match_data)
                    print(f"✔ {home_team} {home_score}-{away_score} {away_team}")
            except Exception as e:
                print("⚠ Bỏ qua 1 dòng bị lỗi:", e)
    except Exception as e:
        print(f"Lỗi khi crawl {league_name} {season}-{season+1}: {e}")
    finally:
        driver.quit()

    return results

# Cấu hình giải đấu
leagues = [
    {"name": "Premier League", "id": 9, "slug": "Premier-League"},
    {"name": "La Liga", "id": 12, "slug": "La-Liga"},
    {"name": "Serie A", "id": 11, "slug": "Serie-A"},
    {"name": "Bundesliga", "id": 20, "slug": "Bundesliga"},
    {"name": "Ligue 1", "id": 13, "slug": "Ligue-1"},
]

all_matches = []

# Vòng lặp qua tất cả giải và mùa
for league in leagues:
    for season in range(2014, 2023 + 1):  # Từ mùa 2014–2015 đến 2022–2023
        matches = crawl_fbref_match_results(
            league_name=league["name"],
            league_id=league["id"],
            league_slug=league["slug"],
            season=season
        )
        all_matches.extend(matches)

# Lưu toàn bộ dữ liệu
df = pd.DataFrame(all_matches)
df.to_csv("match_results_fbref.csv", index=False)
print(f"\nĐã lưu {len(df)} trận đấu từ 5 giải vào match_results_fbref.csv")
