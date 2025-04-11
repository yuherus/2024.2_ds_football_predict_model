# Hướng Dẫn Crawl Dữ Liệu Bóng Đá

## Giới Thiệu

Tài liệu này cung cấp hướng dẫn chi tiết về cách crawl dữ liệu bóng đá từ hai nguồn chính:
- **FBref**: Cho thống kê trận đấu, bảng xếp hạng và lịch thi đấu
- **Transfermarkt**: Cho giá trị cầu thủ và đội hình


Các giải cần lấy dữ liệu
- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

Mùa giải: Từ mùa 2014-2025 đến mùa 2023-2024

## Yêu Cầu và Cài Đặt

### Thư viện cần thiết
```
pip install requirements.txt
```

### Cấu trúc dự án

```
ds_project/
├── craw_data.py       # Script chính để crawl dữ liệu
├── main.py            # Điểm vào chính của ứng dụng
├── utils/
│   ├── fbref.py       # Module xử lý dữ liệu từ FBref
│   └── transfermarkt.py # Module xử lý dữ liệu từ Transfermarkt
└── data/
    ├── match_results.csv  # Dữ liệu kết quả trận đấu
    ├── standings.csv      # Dữ liệu bảng xếp hạng
    ├── fixtures.csv       # Dữ liệu lịch thi đấu
    └── player_values.csv  # Dữ liệu giá trị cầu thủ
```

## 1. Crawl Dữ Liệu từ FBref

FBref là nguồn dữ liệu chính cho thống kê trận đấu, bảng xếp hạng và lịch thi đấu.

### 1.1 Kết Quả Trận Đấu và Thống Kê

#### URL mẫu

```
https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures
```


#### Code mẫu

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def get_match_results(league_id=9, season="2023-2024"):
    """
    Crawl kết quả trận đấu từ FBref
    
    Args:
        league_id: Mã giải đấu (mặc định: 9 - Premier League)
        season: Mùa giải (mặc định: 2023-2024)
    
    Returns:
        DataFrame chứa kết quả trận đấu
    """
    url = f"https://fbref.com/en/comps/{league_id}/schedule/{season}-Premier-League-Scores-and-Fixtures"
    
    # Thêm User-Agent để tránh bị chặn
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Lỗi khi truy cập URL: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tìm bảng lịch thi đấu
    table = soup.find('table', id='sched_all')
    
    if not table:
        print("Không tìm thấy bảng lịch thi đấu")
        return None
    
    # Lấy dữ liệu từ bảng
    matches = []
    
    for row in table.find_all('tr')[1:]:  # Bỏ qua hàng tiêu đề
        cells = row.find_all(['th', 'td'])
        
        # Kiểm tra nếu là hàng có dữ liệu trận đấu
        if len(cells) > 9:
            match_data = {
                'date': cells[0].text.strip(),
                'time': cells[1].text.strip(),
                'home_team': cells[2].text.strip(),
                'score': cells[3].text.strip(),
                'away_team': cells[4].text.strip(),
                'venue': cells[5].text.strip(),
                'competition': cells[6].text.strip(),
            }
            
            # Kiểm tra nếu trận đấu đã diễn ra (có kết quả)
            if match_data['score'] and match_data['score'] != '':
                # Lấy URL chi tiết trận đấu nếu có
                match_url_element = cells[3].find('a')
                if match_url_element and 'href' in match_url_element.attrs:
                    match_url = "https://fbref.com" + match_url_element['href']
                    match_data['match_url'] = match_url
                    
                    # Crawl thống kê chi tiết trận đấu
                    match_stats = get_match_stats(match_url)
                    if match_stats:
                        match_data.update(match_stats)
            
            matches.append(match_data)
    
    # Tạo DataFrame
    df = pd.DataFrame(matches)
    return df

def get_match_stats(match_url):
    """
    Crawl thống kê chi tiết trận đấu
    
    Args:
        match_url: URL chi tiết trận đấu
    
    Returns:
        Dictionary chứa thống kê trận đấu
    """
    # Thêm delay ngẫu nhiên để tránh bị chặn
    time.sleep(random.uniform(1, 3))
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(match_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Lỗi khi truy cập URL chi tiết trận đấu: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tìm bảng thống kê
    stats_table = soup.find('div', id='team_stats')
    
    if not stats_table:
        print("Không tìm thấy bảng thống kê trận đấu")
        return None
    
    # Lấy thông số trận đấu
    stats = {}
    
    # Kiểm soát bóng
    possession = stats_table.find_all('div', class_='possession')
    if possession and len(possession) >= 2:
        stats['home_possession'] = possession[0].text.strip()
        stats['away_possession'] = possession[1].text.strip()
    
    # Các thông số khác
    for stat_div in stats_table.find_all('div', class_='stat_group'):
        for stat in stat_div.find_all('div'):
            stat_text = stat.text.strip()
            
            # Xử lý các thông số như sút, sút trúng đích, v.v.
            if 'Shots' in stat_text:
                values = stat.find_all('div', class_='statistic')
                if len(values) >= 2:
                    stats['home_shots'] = values[0].text.strip()
                    stats['away_shots'] = values[1].text.strip()
            
            elif 'Shots on Target' in stat_text:
                values = stat.find_all('div', class_='statistic')
                if len(values) >= 2:
                    stats['home_shots_on_target'] = values[0].text.strip()
                    stats['away_shots_on_target'] = values[1].text.strip()
            
            elif 'Pass Completion' in stat_text:
                values = stat.find_all('div', class_='statistic')
                if len(values) >= 2:
                    stats['home_pass_completion'] = values[0].text.strip()
                    stats['away_pass_completion'] = values[1].text.strip()
    
    # Lấy thông tin thẻ phạt, phạt góc, v.v.
    events_table = soup.find('div', id='events_wrap')
    if events_table:
        # Đếm thẻ vàng, thẻ đỏ
        home_yellow = len(events_table.find_all('div', class_='event_icon yellow_card home'))
        away_yellow = len(events_table.find_all('div', class_='event_icon yellow_card away'))
        home_red = len(events_table.find_all('div', class_='event_icon red_card home'))
        away_red = len(events_table.find_all('div', class_='event_icon red_card away'))
        
        stats['home_yellow_cards'] = home_yellow
        stats['away_yellow_cards'] = away_yellow
        stats['home_red_cards'] = home_red
        stats['away_red_cards'] = away_red
    
    # Lấy đội hình ra sân
    lineups = soup.find('div', id='lineups')
    if lineups:
        home_lineup = []
        away_lineup = []
        
        home_starters = lineups.find('div', id='a')
        away_starters = lineups.find('div', id='b')
        
        if home_starters:
            for player in home_starters.find_all('tr'):
                player_name = player.find('th')
                if player_name:
                    home_lineup.append(player_name.text.strip())
        
        if away_starters:
            for player in away_starters.find_all('tr'):
                player_name = player.find('th')
                if player_name:
                    away_lineup.append(player_name.text.strip())
        
        stats['home_lineup'] = ', '.join(home_lineup)
        stats['away_lineup'] = ', '.join(away_lineup)
    
    # Lấy thông tin cầu thủ vắng mặt
    missing_players = soup.find('div', id='missing')
    if missing_players:
        home_missing = []
        away_missing = []
        
        teams = missing_players.find_all('div', class_='table_wrapper')
        if len(teams) >= 2:
            for player in teams[0].find_all('tr')[1:]:  # Bỏ qua hàng tiêu đề
                player_name = player.find('th')
                if player_name:
                    home_missing.append(player_name.text.strip())
            
            for player in teams[1].find_all('tr')[1:]:  # Bỏ qua hàng tiêu đề
                player_name = player.find('th')
                if player_name:
                    away_missing.append(player_name.text.strip())
        
        stats['home_missing'] = ', '.join(home_missing)
        stats['away_missing'] = ', '.join(away_missing)
    
    return stats
```

#### Thông số cần crawl

- Vòng(week)
- Sân nhà (home), sân khách(away)
- Đội hình ra sân (home_lineup, away_lineup)
- Vắng mặt (home_missing, away_missing)
- Kiểm soát bóng (home_possession, away_possession)
- Sút (home_shots, away_shots)
- Sút trúng đích (home_shots_on_target, away_shots_on_target)
- Chuyền thành công (home_pass_completion, away_pass_completion)
- Thẻ đỏ (home_red_cards, away_red_cards)
- Thẻ vàng (home_yellow_cards, away_yellow_cards)
- Cản phá (home_saves, away_saves)
- Lỗi (home_fouls, away_fouls)
- Phạt góc (home_corners, away_corners)

### 1.2 Bảng Xếp Hạng

#### URL mẫu

```
https://fbref.com/en/comps/9/Premier-League-Stats
```

#### Code mẫu

```python
def get_standings(league_id=9, season="2023-2024"):
    """
    Crawl bảng xếp hạng từ FBref
    
    Args:
        league_id: Mã giải đấu (mặc định: 9 - Premier League)
        season: Mùa giải (mặc định: 2023-2024)
    
    Returns:
        DataFrame chứa bảng xếp hạng
    """
    url = f"https://fbref.com/en/comps/{league_id}/{season}/Premier-League-Stats"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Lỗi khi truy cập URL: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tìm bảng xếp hạng
    table = soup.find('table', id='results2023-202491_overall')
    
    if not table:
        print("Không tìm thấy bảng xếp hạng")
        return None
    
    # Lấy dữ liệu từ bảng
    standings = []
    
    for row in table.find_all('tr')[1:]:  # Bỏ qua hàng tiêu đề
        cells = row.find_all(['th', 'td'])
        
        if len(cells) > 10:
            team_data = {
                'rank': cells[0].text.strip(),
                'team': cells[1].text.strip(),
                'matches_played': cells[2].text.strip(),
                'wins': cells[3].text.strip(),
                'draws': cells[4].text.strip(),
                'losses': cells[5].text.strip(),
                'goals_for': cells[6].text.strip(),
                'goals_against': cells[7].text.strip(),
                'goal_diff': cells[8].text.strip(),
                'points': cells[9].text.strip(),
                'last_5': get_last_5_results(cells[1].find('a')['href']) if cells[1].find('a') else ''
            }
            
            standings.append(team_data)
    
    # Tạo DataFrame
    df = pd.DataFrame(standings)
    return df

def get_last_5_results(team_url):
    """
    Lấy kết quả 5 trận gần nhất của đội bóng
    
    Args:
        team_url: URL trang đội bóng
    
    Returns:
        Chuỗi kết quả 5 trận gần nhất (W-D-L-W-D)
    """
    full_url = f"https://fbref.com{team_url}"
    
    # Thêm delay ngẫu nhiên để tránh bị chặn
    time.sleep(random.uniform(1, 3))
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(full_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Lỗi khi truy cập URL đội bóng: {response.status_code}")
        return ""
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tìm bảng lịch sử trận đấu
    table = soup.find('table', id='matchlogs_for')
    
    if not table:
        return ""
    
    # Lấy kết quả 5 trận gần nhất
    results = []
    
    for row in table.find_all('tr')[1:6]:  # 5 trận gần nhất
        cells = row.find_all(['th', 'td'])
        
        if len(cells) > 7:
            result_cell = cells[7]  # Cột kết quả
            result = result_cell.text.strip()
            
            if result == 'W':
                results.append('W')
            elif result == 'D':
                results.append('D')
            elif result == 'L':
                results.append('L')
    
    return '-'.join(results)
```

#### Thông số cần crawl

- Số trận (matches_played)
- Điểm (points)
- Bàn thắng (goals_for - GF)
- Bàn thua (goals_against - GA)
- Hiệu số (goal_diff - GD)
- Kết quả 5 trận gần nhất (last_5) (này có thể tự tạo ra từ match data)

### 1.3 Lịch Thi Đấu Sắp Tới

#### URL mẫu

```
https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures
```

#### Code mẫu

```python
def get_upcoming_fixtures(league_id=9, season="2023-2024"):
    """
    Crawl lịch thi đấu sắp tới từ FBref
    
    Args:
        league_id: Mã giải đấu (mặc định: 9 - Premier League)
        season: Mùa giải (mặc định: 2023-2024)
    
    Returns:
        DataFrame chứa lịch thi đấu sắp tới
    """
    url = f"https://fbref.com/en/comps/{league_id}/schedule/{season}-Premier-League-Scores-and-Fixtures"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Lỗi khi truy cập URL: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Tìm bảng lịch thi đấu
    table = soup.find('table', id='sched_all')
    
    if not table:
        print("Không tìm thấy bảng lịch thi đấu")
        return None
    
    # Lấy dữ liệu từ bảng
    fixtures = []
    
    for row in table.find_all('tr')[1:]:  # Bỏ qua hàng tiêu đề
        cells = row.find_all(['th', 'td'])
        
        # Kiểm tra nếu là hàng có dữ liệu trận đấu
        if len(cells) > 9:
            match_data = {
                'date': cells[0].text.strip(),
                'time': cells[1].text.strip(),
                'home_team': cells[2].text.strip(),
                'score': cells[3].text.strip(),
                'away_team': cells[4].text.strip(),
                'venue': cells[5].text.strip(),
                'competition': cells[6].text.strip(),
            }
            
            # Chỉ lấy các trận chưa diễn ra (không có kết quả)
            if not match_data['score'] or match_data['score'] == '':
                fixtures.append(match_data)
    
    # Tạo DataFrame
    df = pd.DataFrame(fixtures)
    return df
```

#### Thông số cần crawl

- Vòng(week)
- Sân nhà (home), sân khách(away)

## 2. Crawl Dữ Liệu từ Transfermarkt

Transfermarkt là nguồn dữ liệu chính cho giá trị cầu thủ và đội hình.

### 2.1 Giá Trị Cầu Thủ

#### URL mẫu

```
https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1
```
#### Code mẫu

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def get_player_values(league_code="GB1"):
    """
    Crawl giá trị cầu thủ từ Transfermarkt
    
    Args:
        league_code: Mã giải đấu (mặc định: GB1 - Premier League)
    
    Returns:
        DataFrame chứa giá trị cầu thủ
    """
    url = f"https://www.transfermarkt.com/premier-league/startseite/wettbewerb/{league_code}"
    
    # Cấu hình Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Chạy ẩn
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        
        # Đợi trang tải xong
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "items"))
        )
        
        # Lấy danh sách đội bóng
        teams_table = driver.find_element(By.CLASS_NAME, "items")
        team_links = teams_table.find_elements(By.CSS_SELECTOR, "a.vereinprofil_tooltip")
        
        team_urls = []
        for link in team_links:
            href = link.get_attribute("href")
            if "/startseite/verein/" in href and href not in team_urls:
                team_urls.append(href)
        
        all_players = []
        
        # Duyệt qua từng đội bóng
        for team_url in team_urls:
            # Thêm delay ngẫu nhiên để tránh bị chặn
            time.sleep(random.uniform(2, 5))
            
            driver.get(team_url)
            
            # Đợi trang tải xong
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "yw1"))
            )
            
            # Lấy tên đội bóng
            team_name = driver.find_element(By.CLASS_NAME, "data-header__headline-wrapper").text.strip()
            
            # Lấy danh sách cầu thủ
            players_table = driver.find_element(By.ID, "yw1")
            player_rows = players_table.find_elements(By.CSS_SELECTOR, "tr.odd, tr.even")
            
            for row in player_rows:
                try:
                    # Lấy tên cầu thủ
                    player_name = row.find_element(By.CSS_SELECTOR, "td.hauptlink a").text.strip()
                    
                    # Lấy vị trí
                    position = row.find_element(By.CSS_SELECTOR, "td:nth-child(2)").text.strip()
                    
                    # Lấy tuổi
                    age = row.find_element(By.CSS_SELECTOR, "td.zentriert:nth-child(3)").text.strip()
                    
                    # Lấy quốc tịch
                    nationality = row.find_element(By.CSS_SELECTOR, "td.zentriert:nth-child(4) img").get_attribute("title")
                    
                    # Lấy giá trị cầu thủ
                    value_element = row.find_element(By.CSS_SELECTOR, "td.rechts")
                    value = value_element.text.strip()
                    
                    player_data = {
                        'player_name': player_name,
                        'team': team_name,
                        'position': position,
                        'age': age,
                        'nationality': nationality,
                        'value': value
                    }
                    
                    all_players.append(player_data)
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý cầu thủ: {e}")
                    continue
        
        # Tạo DataFrame
        df = pd.DataFrame(all_players)
        return df
        
    finally:
        driver.quit()
```
#### Thông số cần crawl
- Giá trị đội bóng
- Giá trị từng cầu thủ trong đội bóng
