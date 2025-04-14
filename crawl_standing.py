import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def crawl_standings(driver, league, season):
    print(f"Crawling standings for {league} season {season}...")
    standings = []

    try:
        if league == "premierleague":
            url = f"https://www.transfermarkt.com/premier-league/tabelle/wettbewerb/GB1?saison_id={season}"
        elif league == "bundesliga":
            url = f"https://www.transfermarkt.com/bundesliga/tabelle/wettbewerb/L1?saison_id={season}"
        elif league == "laliga":
            url = f"https://www.transfermarkt.com/laliga/tabelle/wettbewerb/ES1?saison_id={season}"
        elif league == "seriea":
            url = f"https://www.transfermarkt.com/serie-a/tabelle/wettbewerb/IT1?saison_id={season}"
        elif league == "ligue1":
            url = f"https://www.transfermarkt.com/ligue-1/tabelle/wettbewerb/FR1?saison_id={season}"
        else:
            print(f"Không hỗ trợ giải đấu: {league}")
            return []

        print(f"Đang truy cập {url}...")
        driver.get(url)
        time.sleep(10)

        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.items tbody tr")

        for row in rows:
            try:
                club_element = row.find_elements(By.CSS_SELECTOR, "td.no-border-links.hauptlink a")
                club = club_element[0].text if club_element else "N/A"

                cols = row.find_elements(By.CSS_SELECTOR, "td.zentriert:not(.no-border-rechts)")

                matches_played = cols[0].text.strip()
                wins = cols[1].text.strip()
                draws = cols[2].text.strip()
                losses = cols[3].text.strip()

                goals = cols[4].text.strip()
                goals_for, goals_against = map(int, goals.split(":"))
                goals_diff = cols[5].text.strip()
                points = cols[6].text.strip()

                data = {
                    "club": club,
                    "matches_played": matches_played,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goals_diff": goals_diff,
                    "points": points,
                    "league": league,
                    "season": season
                }
                print(f"{data}")
                
                standings.append(data)
            except Exception as e:
                print(f"Lỗi xử lý hàng: {e}")
                continue

    except Exception as e:
        print(f"Lỗi khi crawl standings: {e}")

    return standings

# Cấu hình giải đấu và mùa giải
leagues = ["premierleague", "bundesliga", "laliga", "seriea", "ligue1"]
seasons = list(range(2014, 2025))
# leagues = ["premierleague"]
# seasons = [2022, 2023]

all_standings = []

driver = setup_driver()

try:
  for league in leagues:
      for season in seasons:
          standings = crawl_standings(driver, league, season)
          all_standings.extend(standings)
          time.sleep(5)

finally:
    driver.quit()

# Lưu vào file CSV
pd.DataFrame(all_standings).to_csv("standings.csv", index=False)
