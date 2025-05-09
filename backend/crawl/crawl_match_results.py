import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless")
    return webdriver.Chrome(options=chrome_options)

def crawl_match_details(driver, url):
    print("url", url)
    details = {
        "round": None, "venue": None,
        "home_lineup": None, "away_lineup": None,
        "home_missing": None, "away_missing": None,
        "home_possession": None, "away_possession": None,
        "home_shots": None, "away_shots": None,
        "home_shots_on_target": None, "away_shots_on_target": None,
        "home_pass_completion": None, "away_pass_completion": None,
        "home_red_cards": None, "away_red_cards": None,
        "home_yellow_cards": None, "away_yellow_cards": None,
        "home_saves": None, "away_saves": None,
        "home_fouls": None, "away_fouls": None,
        "home_corners": None, "away_corners": None,
    }

    try:
        driver.get(url)
        time.sleep(3)

        # Thông tin chung
        try:
            meta = driver.find_element(By.CSS_SELECTOR, "div.scorebox_meta").text.split("\n")
            for line in meta:
                if "Matchweek" in line:
                    details["round"] = line.split("Matchweek")[1].strip().split(")")[0]
                if "Venue:" in line:
                    details["venue"] = line.replace("Venue:", "").strip()
        except:
            pass
        # Thống kê trận đấu
        try:
            team_stats = driver.find_element(By.CSS_SELECTOR, "#team_stats")
            stats_rows = team_stats.find_elements(By.CSS_SELECTOR, "tbody tr")
            
            current_stat = None
            for row in stats_rows:
                try:
                    th_elements = row.find_elements(By.TAG_NAME, "th")
                    if len(th_elements) == 2:
                        continue
                    elif len(th_elements) == 1:
                        current_stat = th_elements[0].text.strip().lower()
                        continue
                    print("current_stat", current_stat)
                    if current_stat:
                        tds = row.find_elements(By.TAG_NAME, "td")
                        if len(tds) == 2:
                            if "cards" in current_stat:
                                try:
                                    home_yellows = len(tds[0].find_elements(By.CSS_SELECTOR, "span.yellow_card"))
                                    away_yellows = len(tds[1].find_elements(By.CSS_SELECTOR, "span.yellow_card"))
                                    home_reds = len(tds[0].find_elements(By.CSS_SELECTOR, "span.red_card"))
                                    away_reds = len(tds[1].find_elements(By.CSS_SELECTOR, "span.red_card"))
                                    
                                    details["home_yellow_cards"] = str(home_yellows)
                                    details["away_yellow_cards"] = str(away_yellows)
                                    details["home_red_cards"] = str(home_reds)
                                    details["away_red_cards"] = str(away_reds)
                                except Exception as e:
                                    print(f"Error processing cards: {e}")
                            else:
                                try:
                                    home_div = tds[0].find_element(By.CSS_SELECTOR, "div div strong")
                                    away_div = tds[1].find_element(By.CSS_SELECTOR, "div div strong")
                                    
                                    if "possession" in current_stat:
                                        details["home_possession"] = home_div.text.replace("%", "")
                                        details["away_possession"] = away_div.text.replace("%", "")
                                    elif "passing accuracy" in current_stat:
                                        details["home_pass_completion"] = home_div.text.replace("%", "")
                                        details["away_pass_completion"] = away_div.text.replace("%", "")
                                    elif "shots on target" in current_stat:
                                        try:
                                            # Parse home shots (format: "5 of 14 — 36%")
                                            home_text = tds[0].find_element(By.CSS_SELECTOR, "div div").text
                                            home_parts = home_text.split(" of ")
                                            details["home_shots_on_target"] = home_parts[0].strip()
                                            details["home_shots"] = home_parts[1].split("—")[0].strip()
                                            
                                            # Parse away shots (format: "80% — 4 of 5")
                                            away_text = tds[1].find_element(By.CSS_SELECTOR, "div div").text
                                            away_parts = away_text.split("—")[1].strip()
                                            away_numbers = away_parts.split(" of ")
                                            details["away_shots_on_target"] = away_numbers[0].strip()
                                            details["away_shots"] = away_numbers[1].strip()
                                            
                                        except Exception as e:
                                            print(f"Error processing shots data: {e}")
                                            
                                    elif "saves" in current_stat:
                                        try:
                                            # Parse home saves
                                            home_text = tds[0].find_element(By.CSS_SELECTOR, "div div").text
                                            home_parts = home_text.split(" of ")
                                            details["home_saves"] = home_parts[0].strip()
                                            
                                            # Parse away saves
                                            away_text = tds[1].find_element(By.CSS_SELECTOR, "div div").text
                                            away_parts = away_text.split(" of ")
                                            details["away_saves"] = away_parts[0].strip()
                                            
                                        except Exception as e:
                                            print(f"Error processing saves data: {e}")
                                except Exception as e:
                                    print(f"Error processing stat {current_stat}: {e}")
                except Exception as e:
                    print(f"Error processing team stats row: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error extracting team stats: {e}")
            pass

        try:
            extra_stats = driver.find_element(By.CSS_SELECTOR, "#team_stats_extra")
            stats_divs = extra_stats.find_elements(By.TAG_NAME, "div")
            
            for stats_div in stats_divs:
                try:
                    all_divs = stats_div.find_elements(By.TAG_NAME, "div")
                    if len(all_divs) < 6:  # Skip header sections
                        continue
                        
                    for i in range(0, len(all_divs)-2, 3):
                        home_val = all_divs[i].text.strip()
                        label = all_divs[i+1].text.strip().lower()
                        away_val = all_divs[i+2].text.strip()
                        
                        if "fouls" in label:
                            details["home_fouls"] = home_val
                            details["away_fouls"] = away_val
                        elif "corners" in label:
                            details["home_corners"] = home_val
                            details["away_corners"] = away_val
                        elif "shots" in label and details["home_shots"] is None:
                            details["home_shots"] = home_val
                            details["away_shots"] = away_val
                except Exception as e:
                    print(f"Error processing extra stats div: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error extracting extra stats: {e}")
            pass

        # Fix shots on target and saves values
        try:
            if details["home_shots_on_target"] and "—" in details["home_shots_on_target"]:
                details["home_shots_on_target"] = details["home_shots_on_target"].split("—")[0].strip()
            if details["away_shots_on_target"] and "—" in details["away_shots_on_target"]:
                details["away_shots_on_target"] = details["away_shots_on_target"].split("—")[0].strip()
            if details["home_saves"] and "—" in details["home_saves"]:
                details["home_saves"] = details["home_saves"].split("—")[0].strip()
            if details["away_saves"] and "—" in details["away_saves"]:
                details["away_saves"] = details["away_saves"].split("—")[0].strip()
        except Exception as e:
            print(f"Error fixing shot/save values: {e}")
            pass

        # Đội hình
        try:
            home_lineup = driver.find_element(By.CSS_SELECTOR, "#a table tbody")
            away_lineup = driver.find_element(By.CSS_SELECTOR, "#b table tbody")
            
            # Get home team formation and players
            home_formation = home_lineup.find_element(By.TAG_NAME, "th").text.split("(")[1].strip(")")
            home_players = []
            for row in home_lineup.find_elements(By.TAG_NAME, "tr")[1:12]:
                tds = row.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 2:  # Make sure we have at least 2 td elements
                    player_name = tds[1].text.strip()  # Get text from second td which contains player name
                    if player_name:
                        home_players.append(player_name)
            
            # Get away team formation and players
            away_formation = away_lineup.find_element(By.TAG_NAME, "th").text.split("(")[1].strip(")")
            away_players = []
            for row in away_lineup.find_elements(By.TAG_NAME, "tr")[1:12]:
                tds = row.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 2:
                    player_name = tds[1].text.strip()
                    if player_name:
                        away_players.append(player_name)
            details["home_lineup"] = f"{home_formation}: {', '.join(home_players)}"
            details["away_lineup"] = f"{away_formation}: {', '.join(away_players)}"
            
            # Get bench players (missing/substitutes)
            home_bench = []
            for row in home_lineup.find_elements(By.TAG_NAME, "tr")[13:]:  # After "Bench" header
                tds = row.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 2:
                    player_name = tds[1].text.strip()
                    if player_name:
                        home_bench.append(player_name)
                        
            away_bench = []
            for row in away_lineup.find_elements(By.TAG_NAME, "tr")[13:]:  # After "Bench" header
                tds = row.find_elements(By.TAG_NAME, "td")
                if len(tds) >= 2:
                    player_name = tds[1].text.strip()
                    if player_name:
                        away_bench.append(player_name)
            
            details["home_missing"] = ", ".join(home_bench)
            details["away_missing"] = ", ".join(away_bench)
        except Exception as e:
            print(f"Error extracting lineup: {e}")
            pass

    except Exception as e:
        print(f"⚠ Không thể lấy chi tiết từ {url}: {e}")

    return details

def get_match_list(driver, main_url, league_name, season):
    matches = []
    try:
        driver.get(main_url)
        time.sleep(5)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.stats_table tbody tr")
        print(f"Found {len(rows)} matches")
        
        for row in tqdm(rows, desc="Getting match list"):
            try:
                row_class = row.get_attribute("class")
                if row_class == "thead":
                    continue
                
                # Get basic match data
                row_data = {
                    'date': row.find_element(By.CSS_SELECTOR, "td[data-stat='date']").text,
                    'home_team': row.find_element(By.CSS_SELECTOR, "td[data-stat='home_team']").text,
                    'away_team': row.find_element(By.CSS_SELECTOR, "td[data-stat='away_team']").text,
                    'score': row.find_element(By.CSS_SELECTOR, "td[data-stat='score']").text
                }
                
                # Get match report URL
                report_link = row.find_element(By.CSS_SELECTOR, "td[data-stat='match_report'] a")
                match_report_url = report_link.get_attribute("href")
                
                if row_data['score'] and "–" in row_data['score']:
                    home_score, away_score = map(int, row_data['score'].split("–"))
                    match_data = {
                        "date": row_data['date'],
                        "home_team": row_data['home_team'],
                        "away_team": row_data['away_team'],
                        "home_score": home_score,
                        "away_score": away_score,
                        "league": league_name.lower().replace(" ", ""),
                        "season": f"{season}-{season+1}",
                        "source": "FBref",
                        "match_report_url": match_report_url
                    }
                    matches.append(match_data)
                    
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
                
    except Exception as e:
        print(f"Error getting match list: {e}")
    
    return matches

def crawl_fbref_match_results(league_name, league_id, league_slug, season):
    print(f"\n🟢 Crawling {league_name} - {season}-{season+1}...")
    driver = setup_driver()
    results = []

    try:
        main_url = f"https://fbref.com/en/comps/{league_id}/{season}-{season+1}/schedule/{season}-{season+1}-{league_slug}-Scores-and-Fixtures"
        
        # Step 1: Get all matches basic data and their report URLs
        matches = get_match_list(driver, main_url, league_name, season)
        print(f"Found {len(matches)} matches to process")
        
        # Step 2: Get detailed information for each match
        for match in tqdm(matches, desc="Processing match details"):
            try:
                details = crawl_match_details(driver, match['match_report_url'])
                match.update(details)
                results.append(match)
                print(f"✔ {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
            except Exception as e:
                print(f"Error processing match details: {e}")
                continue
                
    except Exception as e:
        print(f"Error in crawling process: {e}")
    finally:
        driver.quit()

    return results

# Danh sách các giải đấu
leagues = [
    {"name": "Premier League", "id": 9, "slug": "Premier-League"},
    {"name": "La Liga", "id": 12, "slug": "La-Liga"},
    {"name": "Serie A", "id": 11, "slug": "Serie-A"},
    {"name": "Bundesliga", "id": 20, "slug": "Bundesliga"},
    {"name": "Ligue 1", "id": 13, "slug": "Ligue-1"},
]

all_matches = []

# Vòng lặp qua các mùa và giải
for league in leagues:
    for season in range(2014, 2015):  # Từ mùa 2014–2015 đến 2023–2024
        matches = crawl_fbref_match_results(
            league_name=league["name"],
            league_id=league["id"],
            league_slug=league["slug"],
            season=season
        )
        print("matches", matches)
        all_matches.extend(matches)

# Lưu dữ liệu ra CSV
df = pd.DataFrame(all_matches)
df.to_csv("match_results_fbref_detailed.csv", index=False)
print(f"\n✅ Đã lưu {len(df)} trận đấu vào match_results_fbref_detailed.csv")
