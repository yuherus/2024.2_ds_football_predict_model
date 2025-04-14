import random
import os
import traceback
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


def setup_driver():
    """Setup WebDriver with explicit path for WSL2 environment."""
    chrome_options = Options()

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")

    # Path to the manually downloaded ChromeDriver
    chromedriver_path = os.path.expanduser("~/chromedriver/chromedriver-linux64/chromedriver")

    # Ensure the path exists
    if not os.path.exists(chromedriver_path):
        raise FileNotFoundError(f"ChromeDriver not found at {chromedriver_path}. Please download it first.")

    # Create service with explicit path
    service = Service(executable_path=chromedriver_path)

    # Initialize Chrome
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def crawl_player_values(league, season):
    """Crawl giá trị cầu thủ từ Transfermarkt using the marktwerte (market values) page."""
    print(f"Crawling player values for {league} season {season}...")
    driver = setup_driver()
    player_values = []

    # Mapping leagues to their correct transfermarkt slugs and codes
    league_mapping = {
        "premierleague": {"slug": "premier-league", "code": "GB1", "name": "Premier League"},
        "bundesliga": {"slug": "bundesliga", "code": "L1", "name": "Bundesliga"},
        "laliga": {"slug": "laliga", "code": "ES1", "name": "LaLiga"},
        "seriea": {"slug": "serie-a", "code": "IT1", "name": "Serie A"},
        "ligue1": {"slug": "ligue-1", "code": "FR1", "name": "Ligue 1"}
    }

    try:
        # Get league information
        league_info = league_mapping.get(league, {"slug": "premier-league", "code": "GB1", "name": "Premier League"})
        league_slug = league_info["slug"]
        league_code = league_info["code"]
        league_name = league_info["name"]

        # Use the direct market values page which has all players listed
        base_url = f"https://www.transfermarkt.com/{league_slug}/marktwerte/wettbewerb/{league_code}/plus/1"

        # Track pages processed
        current_page = 1
        has_next_page = True

        while has_next_page and current_page <= 10:
            # Construct URL with season parameter and pagination
            if current_page == 1:
                url = f"{base_url}?saison_id={season}"
            else:
                url = f"{base_url}/page/{current_page}?saison_id={season}"

            print(f"Processing page {current_page}: {url}")
            driver.get(url)

            # Longer wait time to ensure page loads properly
            time.sleep(10)

            # Save HTML for debugging the first page
            if current_page == 1:
                with open(f"{league}_{season}_debug_page.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)

            # Direct JavaScript approach to extract data (more reliable)
            player_data_script = """
            const players = [];
            const rows = document.querySelectorAll('table.items > tbody > tr.odd, table.items > tbody > tr.even');

            rows.forEach(row => {
                try {
                    // Extract player name
                    const playerNameElem = row.querySelector('td table.inline-table td.hauptlink a');
                    const playerName = playerNameElem ? playerNameElem.textContent.trim() : "Unknown";

                    // Extract position
                    const positionElem = row.querySelector('td table.inline-table tr:nth-child(2) td');
                    const position = positionElem ? positionElem.textContent.trim() : "Unknown";

                    // Extract nationality
                    const flagElems = row.querySelectorAll('img.flaggenrahmen');
                    const nationalities = [];
                    flagElems.forEach(flag => {
                        const nat = flag.getAttribute('title');
                        if (nat && !nationalities.includes(nat)) {
                            nationalities.push(nat);
                        }
                    });
                    const nationality = nationalities.length > 0 ? nationalities.join(' / ') : "Unknown";

                    // Extract age
                    const ageTds = Array.from(row.querySelectorAll('td.zentriert')).filter(td => 
                        !td.querySelector('table') && !td.querySelector('a[href*="/verein/"]')
                    );
                    const age = ageTds.length > 0 ? ageTds[0].textContent.trim() : "Unknown";

                    // Extract team
                    const teamElem = row.querySelector('td.zentriert a[href*="/verein/"]');
                    let team = "Unknown";
                    if (teamElem) {
                        const teamImg = teamElem.querySelector('img');
                        if (teamImg) {
                            team = teamImg.getAttribute('alt') || teamImg.getAttribute('title') || "Unknown";
                            if (team === "" || team === "&nbsp;") {
                                const href = teamElem.getAttribute('href');
                                if (href && href.includes('/verein/')) {
                                    const parts = href.split('/');
                                    for (let i = 0; i < parts.length; i++) {
                                        if (parts[i] === 'verein' && i > 0) {
                                            team = parts[i-1].replace(/-/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Extract market value
                    let value = "Unknown";
                    const valueElems = row.querySelectorAll('td.rechts');
                    for (const elem of valueElems) {
                        if (elem.textContent.includes('€')) {
                            value = elem.textContent.trim();
                            break;
                        }
                    }

                    players.push({
                        player_name: playerName,
                        position: position,
                        nationality: nationality,
                        age: age,
                        team: team,
                        value: value
                    });
                } catch (e) {
                    console.error('Error processing row:', e);
                }
            });

            return JSON.stringify(players);
            """

            # Execute JavaScript to extract data
            result = driver.execute_script(player_data_script)

            import json
            extracted_players = json.loads(result)
            print(f"JavaScript extracted {len(extracted_players)} players")

            # Process the extracted data
            for player_data in extracted_players:
                player_data['league'] = league_name
                player_data['season'] = season
                player_values.append(player_data)

                print(
                    f"Added player: {player_data['player_name']}, {player_data['position']}, Value: {player_data['value']}")

            # Check if there's a next page using JavaScript
            has_next_page_script = """
            const nextButton = document.querySelector('li.tm-pagination__list-item--icon-next-page');
            return nextButton !== null;
            """

            has_next_page = driver.execute_script(has_next_page_script)

            if has_next_page and current_page < 10:
                current_page += 1
                # Wait between pages to avoid being blocked
                delay = random.uniform(3, 5)
                print(f"Waiting {delay:.1f}s before loading next page...")
                time.sleep(delay)
            else:
                has_next_page = False
                print("No more pages or reached page limit")

    except Exception as e:
        print(f"Error crawling player values: {e}")
        traceback.print_exc()

    finally:
        driver.quit()

    print(f"Collected data for {len(player_values)} players from {league} {season}")
    return player_values


leagues = ["premierleague", "laliga", "seriea", "bundesliga", "ligue1"]
seasons = list(range(2014, 2024))  # 2014–2023 -> mùa 2014–2015 đến 2023–2024

all_player_values = []

for league in leagues:
    for season in seasons:
        print(f"Đang crawl dữ liệu {league} mùa {season}...")

        player_values = crawl_player_values(league, season)
        all_player_values.extend(player_values)

        # Tránh bị chặn
        time.sleep(10)

pd.DataFrame(all_player_values).to_csv("player_values.csv", index=False)