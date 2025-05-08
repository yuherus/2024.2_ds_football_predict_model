import random
import os
import traceback
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
from datetime import datetime


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


def crawl_team_with_requests(team_url):
    """Crawl team data using requests and BeautifulSoup"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Referer': 'https://www.transfermarkt.com/'
    }

    print(f"Fetching {team_url}")
    response = requests.get(team_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {team_url}: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract team name
    team_name = soup.select_one('h1.data-header__headline-wrapper')
    if team_name:
        team_name = team_name.text.strip()
    else:
        # Try alternate location
        team_name_alt = soup.select_one('div.dataMain div.dataName')
        team_name = team_name_alt.text.strip() if team_name_alt else "Unknown Team"

    # Extract league and season
    league_name = "Unknown"
    league_elem = soup.select_one('div.data-header__club-info a')
    if league_elem:
        league_name = league_elem.text.strip()

    season_text = "Unknown"
    season_elem = soup.select_one('div.data-header__season-selector div.tm-tabs__option--active')
    if season_elem:
        season_text = season_elem.text.strip()
    else:
        # Extract season from URL
        if 'saison_id' in team_url:
            season_id = team_url.split('saison_id/')[1].split('/')[0].split('?')[0]
            season_text = season_id

    print(f"Processing team: {team_name} | League: {league_name} | Season: {season_text}")

    player_values = []
    player_rows = soup.select('table.items > tbody > tr.odd, table.items > tbody > tr.even')

    print(f"Found {len(player_rows)} player rows")

    for row in player_rows:
        try:
            # Jersey number
            jersey_number = ""
            number_elem = row.select_one('div.rn_nummer')
            if number_elem:
                jersey_number = number_elem.text.strip()

            # Player name
            player_elem = row.select_one('td.hauptlink a')
            if not player_elem:
                continue

            player_name = player_elem.text.strip()
            # Remove injury/suspension indicators
            if '\xa0' in player_name:
                player_name = player_name.split('\xa0')[0]

            # Player URL for additional details if needed
            player_url = player_elem.get('href', '')
            if player_url and not player_url.startswith('http'):
                player_url = f"https://www.transfermarkt.com{player_url}"

            # Position
            position_elem = row.select_one('table.inline-table tr:nth-child(2) td')
            position = position_elem.text.strip() if position_elem else "Unknown"

            # Age and date of birth
            dob_age = "Unknown"
            age = ""
            dob_elem = row.select_one('td.zentriert:nth-of-type(3)')
            if dob_elem:
                dob_age = dob_elem.text.strip()
                # Extract age from parentheses if present
                if "(" in dob_age and ")" in dob_age:
                    age = dob_age.split("(")[1].split(")")[0]

            # Nationality - get all flags
            nationality = "Unknown"
            flag_elems = row.select('td.zentriert img.flaggenrahmen')
            if flag_elems:
                nationalities = []
                for flag in flag_elems:
                    nat = flag.get('title')
                    if nat and nat not in nationalities:
                        nationalities.append(nat)
                nationality = " / ".join(nationalities) if nationalities else "Unknown"

            # Market value
            value_elem = row.select_one('td.rechts.hauptlink a')
            market_value = value_elem.text.strip() if value_elem else "Unknown"

            # Status (injured, suspended, etc.)
            status = "Active"
            status_elem = row.select_one('span[class*="verletzt"], span[class*="ausfall"]')
            if status_elem and status_elem.get('title'):
                status = status_elem.get('title')

            # Check if player is on loan
            on_loan = False
            loan_from = ""
            loan_elem = row.select_one('span.wechsel-kader-wappen')
            if loan_elem and loan_elem.get('title') and "Joined from" in loan_elem.get('title'):
                on_loan = True
                loan_parts = loan_elem.get('title').split("Joined from ")
                if len(loan_parts) > 1:
                    loan_from = loan_parts[1].split(";")[0]

            # Check if player is captain
            is_captain = bool(row.select_one('span.kapitaenicon-table'))

            player_data = {
                'team': team_name,
                'league': league_name,
                'season': season_text,
                'jersey_number': jersey_number,
                'player_name': player_name,
                'position': position,
                'dob_age': dob_age,
                'age': age,
                'nationality': nationality,
                'market_value': market_value,
                'status': "Injured/Suspended" if status != "Active" else "Active",
                'on_loan': on_loan,
                'loan_from': loan_from,
                'is_captain': is_captain
            }

            player_values.append(player_data)
            print(
                f"Added player: {player_name}, {position}, Age: {age}, Nationality: {nationality}, Value: {market_value}")

        except Exception as e:
            print(f"Error processing player: {e}")
            traceback.print_exc()
            continue

    return player_values


def crawl_league_team_values(league, season):
    """
    Crawl team values and squad information for a specific league and season.

    Args:
        league: League slug (e.g., "premierleague")
        season: Season ID (e.g., 2024)

    Returns:
        List of dictionaries containing team data
    """
    # Mapping leagues to their correct transfermarkt slugs and codes
    league_mapping = {
        "premierleague": {"slug": "premier-league", "code": "GB1", "name": "Premier League"},
        "bundesliga": {"slug": "bundesliga", "code": "L1", "name": "Bundesliga"},
        "laliga": {"slug": "laliga", "code": "ES1", "name": "LaLiga"},
        "seriea": {"slug": "serie-a", "code": "IT1", "name": "Serie A"},
        "ligue1": {"slug": "ligue-1", "code": "FR1", "name": "Ligue 1"}
    }

    try:
        league_info = league_mapping.get(league, {"slug": "premier-league", "code": "GB1", "name": "Premier League"})
        league_slug = league_info["slug"]
        league_code = league_info["code"]

        # URL to get league page
        url = f"https://www.transfermarkt.com/{league_slug}/startseite/wettbewerb/{league_code}?saison_id={season}"

        # Use requests for the league page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Referer': 'https://www.transfermarkt.com/'
        }

        print(f"Fetching league data for {league_info['name']} season {season}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch league page: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find team data rows
        team_rows = soup.select('table.items > tbody > tr')
        print(f"Found {len(team_rows)} teams in {league_info['name']} season {season}")

        team_values = []

        for row in team_rows:
            try:
                # Team name
                team_name_elem = row.select_one('td.hauptlink a')
                if not team_name_elem:
                    continue

                team_name = team_name_elem.text.strip()

                # Squad size
                squad_size_elem = row.select_one('td:nth-of-type(3)')
                squad_size = squad_size_elem.text.strip() if squad_size_elem else "Unknown"

                # Average age
                avg_age_elem = row.select_one('td:nth-of-type(4)')
                avg_age = avg_age_elem.text.strip() if avg_age_elem else "Unknown"

                # Foreigners
                foreigners_elem = row.select_one('td:nth-of-type(5)')
                foreigners = foreigners_elem.text.strip() if foreigners_elem else "Unknown"

                # Average market value
                avg_value_elem = row.select_one('td:nth-of-type(6)')
                avg_market_value = avg_value_elem.text.strip() if avg_value_elem else "Unknown"

                # Total market value
                total_value_elem = row.select_one('td:nth-of-type(7) a')
                total_market_value = total_value_elem.text.strip() if total_value_elem else "Unknown"

                # Champion/Promoted status
                status = "Regular"
                if row.select_one('span.icon-aufsteiger'):
                    status = "Promoted"
                elif row.select_one('img.tabelle-erfolg[title*="Champion"]'):
                    status = "Champion"

                team_data = {
                    'team_name': team_name,
                    'league': league_info['name'],
                    'season': season,
                    'squad_size': squad_size,
                    'average_age': avg_age,
                    'foreigners': foreigners,
                    'average_market_value': avg_market_value,
                    'total_market_value': total_market_value,
                    'status': status,
                    'crawl_date': datetime.now().strftime('%Y-%m-%d')
                }

                team_values.append(team_data)
                print(f"Added team: {team_name}, Total Value: {total_market_value}")

            except Exception as e:
                print(f"Error processing team row: {e}")
                traceback.print_exc()
                continue

        return team_values

    except Exception as e:
        print(f"Error in crawl_league_team_values: {e}")
        traceback.print_exc()
        return []



def crawl_league_teams(league, season):
    """
    Crawl all teams in a league and get player values for each team.
    """
    # Mapping leagues to their correct transfermarkt slugs and codes
    league_mapping = {
        "premierleague": {"slug": "premier-league", "code": "GB1", "name": "Premier League"},
        "bundesliga": {"slug": "bundesliga", "code": "L1", "name": "Bundesliga"},
        "laliga": {"slug": "laliga", "code": "ES1", "name": "LaLiga"},
        "seriea": {"slug": "serie-a", "code": "IT1", "name": "Serie A"},
        "ligue1": {"slug": "ligue-1", "code": "FR1", "name": "Ligue 1"}
    }

    all_players = []

    try:
        league_info = league_mapping.get(league, {"slug": "premier-league", "code": "GB1", "name": "Premier League"})
        league_slug = league_info["slug"]
        league_code = league_info["code"]

        # URL to get all teams in the league
        url = f"https://www.transfermarkt.com/{league_slug}/startseite/wettbewerb/{league_code}?saison_id={season}"

        # Use requests for the league page too
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Referer': 'https://www.transfermarkt.com/'
        }

        print(f"Getting teams for {league_info['name']} season {season}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch league page: {response.status_code}")
            # Debug - save the response content
            with open(f"debug_{league}_{season}.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            return []

        # Debug - save the HTML content for inspection
        with open(f"debug_{league}_{season}.html", "w", encoding="utf-8") as f:
            f.write(response.text)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all team links - try multiple selector patterns
        team_links = []

        # Try different selectors to find team links
        selectors = [
            "a.vereinprofil_tooltip",
            "td.hauptlink a",
            "table.items tbody tr td a[href*='/verein/']"
        ]

        for selector in selectors:
            team_elems = soup.select(selector)
            for team_elem in team_elems:
                href = team_elem.get('href', '')
                if href and "/startseite/verein/" in href:
                    if not href.startswith('http'):
                        href = f"https://www.transfermarkt.com{href}"
                    if href not in team_links:
                        team_links.append(href)

            if team_links:
                print(f"Found {len(team_links)} teams using selector: {selector}")
                break

        if not team_links:
            print(f"No team links found for {league} season {season}")
            return []

        # Process each team
        for i, team_url in enumerate(team_links):
            try:
                print(f"Processing team {i + 1}/{len(team_links)}: {team_url}")
                team_players = crawl_team_with_requests(team_url)
                all_players.extend(team_players)

                # Wait between teams to avoid being blocked
                delay = random.uniform(5, 10)
                print(f"Waiting {delay:.1f}s before next team...")
                time.sleep(delay)

            except Exception as e:
                print(f"Error processing team {team_url}: {e}")
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"Error in crawl_league_teams: {e}")
        traceback.print_exc()

    return all_players
# Example usage:
# 1. Crawl a single team
# players = crawl_team_player_values("https://www.transfermarkt.com/real-madrid/startseite/verein/418/saison_id/2024")
# pd.DataFrame(players).to_csv("real_madrid_players.csv", index=False)

# 2. Crawl all teams in a league
# laliga_players = crawl_league_teams("laliga", 2024)
# pd.DataFrame(laliga_players).to_csv("laliga_players_2024.csv", index=False)

# 3. Crawl multiple leagues and seasons
leagues = ["premierleague", "laliga", "seriea", "bundesliga", "ligue1"]
seasons = list(range(2014, 2025))  # 2014-2015 to 2024-2025 seasons

all_players = []
all_team_values = []

for league in leagues:
    for season in seasons:
        print(f"Processing {league} for season {season}")
        league_team_values = crawl_league_team_values(league, season)
        all_team_values.extend(league_team_values)
        league_players = crawl_league_teams(league, season)
        all_players.extend(league_players)

        # Wait between leagues/seasons to avoid being blocked
        time.sleep(15)

# Save all data
pd.DataFrame(all_players).to_csv("all_player_values.csv", index=False)
pd.DataFrame(all_team_values).to_csv("all_team_values.csv", index=False)

# players = crawl_team_with_requests("https://www.transfermarkt.com/real-madrid/startseite/verein/418/saison_id/2024")
# pd.DataFrame(players).to_csv("test.csv", index=False)

        # print(f"Processing {"premierleague"} for season {2014}")
# league_team_values = crawl_league_team_values("premierleague", 2014)
# all_team_values.extend(league_team_values)
# pd.DataFrame(all_team_values).to_csv("all_team_values.csv", index=False)
