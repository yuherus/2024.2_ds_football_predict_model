import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")  # Nếu không muốn mở trình duyệt
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def crawl_fbref_match_results(league="Premier League", season=2022):
    print(f"Crawling FBref data for {league} - {season} season...")
    driver = setup_driver()
    results = []

    try:
        url = f"https://fbref.com/en/comps/9/{season}-{season+1}/schedule/{season}-{season+1}-Premier-League-Scores-and-Fixtures"
        driver.get(url)
        time.sleep(5)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.stats_table tbody tr")
        for row in rows:
            try:
                # Một số dòng là separator nên bỏ qua
                if row.get_attribute("class") == "thead":
                    continue

                date = row.find_element(By.CSS_SELECTOR, "td[data-stat='date']").text
                home_team = row.find_element(By.CSS_SELECTOR, "td[data-stat='home_team']").text
                away_team = row.find_element(By.CSS_SELECTOR, "td[data-stat='away_team']").text
                score = row.find_element(By.CSS_SELECTOR, "td[data-stat='score']").text

                if score and "-" in score:
                    home_score, away_score = map(int, score.split("–"))

                    match_data = {
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "league": "premierleague",
                        "season": season,
                        "source": "FBref"
                    }

                    results.append(match_data)
                    print(f"✔ {home_team} {home_score}-{away_score} {away_team}")
            except Exception as e:
                print("⚠ Bỏ qua 1 dòng bị lỗi:", e)
                continue
    except Exception as e:
        print(f"❌ Lỗi khi crawl FBref: {e}")
    finally:
        driver.quit()

    return results

# Crawl dữ liệu và lưu vào CSV
season = 2022
matches = crawl_fbref_match_results(season=season)
df = pd.DataFrame(matches)
df.to_csv("match_results_fbref.csv", index=False)
print(f"\n✅ Đã lưu {len(df)} trận vào match_results_fbref.csv")
