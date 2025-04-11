import re
import enum
import requests
from bs4 import BeautifulSoup
import pandas as pd

GOLGG_BASE_URL = "https://gol.gg/"
GOLGG_TOURNAMENT_ENDPOINT = "tournament/list/"
GOLGG_TOURNAMENT_SERIES_ENDPOINT = "tournament/tournament-matchlist/"
GOL_GG_BANS_ENDPOINT = "champion/bans-stats/"
GOL_GG_PICKBAN_BY_PATCH_ENDPOINT = "stats/patches-by-patches/"
GOLGG_PHP_URL = "https://gol.gg/tournament/ajax.trlist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
}
HEADER_DETAILED = {
    "Host": "gol.gg",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:136.0) Gecko/20100101 Firefox/136.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://gol.gg",
    "Connection": "keep-alive",
    "Referer": "https://gol.gg/tournament/list/",
    "Cookie": "",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Priority": "u=0"
}

class Split(enum.Enum):
    """
    Enum for split types.
    """
    WINTER = "split-Winter/"
    SUMMER = "split-Summer/"
    SPRING = "split-Spring/"
    ALL = "split-ALL/"

def GOL_GG_GAME_ENDPOINT_GEN(game_code: int) -> str:
    """
    Generates the URL for a game on gol.gg.

    Args:
        game_code (int): The game code.

    Returns:
        str: The URL for the game.
    """
    return f"game/stats/{game_code}/page-game/"

def GOL_GG_SEASON_SPLIT_URL_GEN(season: int, split: Split):
    """
    Generates the URL for a season and split on gol.gg.

    Args:
        season (int): The season.
        split (Split): The split.

    Returns:
        str: The URL for the season and split.
    """
    return f"season-S{season}/{split.value}"

def GOL_GG_PAYLOAD_GEN(season:int) -> str:
    """
    Generates the payload for a season on gol.gg.

    Args:
        season (int): The season.

    Returns:
        str: The payload for the season.
    """
    return {
        "season": f"S{season}",
        "league[]": ["EWC", "First Stand", "IEM", "LCK", "LCP", "LCS", "LEC", "LMS", "LPL", "LTA", 
        "LTA North", "MSC", "MSI", "WORLDS"]
    }

def scrape_pick_ban_by_patch(url: str) -> pd.DataFrame:
    """
    Scrapes pick/ban data from gol.gg for a given season and split.

    Args:
        url (str): The URL to scrape pick/ban data from.

    Returns:
        pd.DataFrame: A DataFrame containing the pick/ban data.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    headers = soup.find_all('th')
    patch_headers = [header.text.strip() for header in headers]

    # Find all table cells
    cells = soup.find_all('td', {'style': 'vertical-align:top'})

    all_champions_data = []

    # Process each cell (corresponds to a patch)
    for i, cell in enumerate(cells):
        patch = patch_headers[i] if i < len(patch_headers) else f"Unknown_Patch_{i}"

        # Find all champion divs in this cell
        champion_divs = cell.find_all('div', {'class': re.compile(r'[A-Za-z]')})

        # Filter out divs that don't contain champion data
        champion_divs = [div for div in champion_divs if div.get('onmouseover') and 'setBg' in div.get('onmouseover')]

        for div in champion_divs:
            # Extract champion name
            champion_name = div.get('class')[0]

            # Extract percentage value
            percentage_text = div.text.strip()
            percentage_match = re.search(r'(\d+)%', percentage_text)
            percentage = int(percentage_match.group(1)) if percentage_match else None

            # Get the champion ID from the URL
            link = div.find('a')
            if link:
                champion_id_match = re.search(r'/champion-stats/(\d+)/', link.get('href'))
                champion_id = champion_id_match.group(1) if champion_id_match else None
            else:
                champion_id = None

            all_champions_data.append({
                'name': champion_name,
                'percentage': percentage,
                'id': champion_id,
                'patch': patch
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_champions_data)
    return df

def get_all_games_from_tournament(html_content: str) -> list[str]:
    """
    Scrapes the URL of all games from a tournament page.

    Args:
        html_content (str): The HTML content of the tournament page.

    Returns:
        list[str]: A list of URLs for all games in the tournament.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    match_links = []

    # Find all 'a' tags within the table containing match information
    table = soup.find('table')
    if table:
        for a_tag in table.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('../game/stats/'):
                full_url = f"https://gol.gg{href[2:]}"
                match_links.append(full_url)
    else:
        print("Match table not found on the page.")

    return match_links
    
def collect_matches_from_game(html_content: str) -> list[str]:
    soup = BeautifulSoup(html_content, 'html.parser')

    #TODO: Find length of series by looking at game menu - return links to each match
    match_links = []

    for link in soup.select(".game-menu-button a.nav-link"):
        href = link.get('href')
        if href and 'page-game' in href:
            match_links.append(f"{GOLGG_BASE_URL}{href.lstrip('../')}")
    return match_links

def scrape_teams_side_winner_from_game(html_content: str) -> pd.DataFrame:
    """
    Scrapes the winner/loser and the team that was blue/red from a game page.

    Args:
        html_content (str): The HTML content of the game page.

    Returns:
        pd.DataFrame: A DataFrame containing the winner/loser and the team that was blue/red.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the winner/loser and the team that was blue/red
    blue_side_div = soup.find('div', class_='col-12 blue-line-header')
    if not blue_side_div:
        return pd.DataFrame()
    blue_side_team = blue_side_div.find('a').text.strip()

    red_side_div = soup.find('div', class_='col-12 red-line-header')
    if not red_side_div:
        return pd.DataFrame()
    red_side_team = red_side_div.find('a').text.strip()

    blue_side_lost = 'LOSS' in blue_side_div.text
    winner = ""
    loser = ""
    if blue_side_lost:
        winner = "red_side"
        loser = "blue_side"
    else:
        winner = "blue_side"
        loser = "red_side"

    temp = {
        "blue_side": blue_side_team,
        "red_side": red_side_team,
        "winner": winner,
        "loser": loser
    }

    return pd.DataFrame([temp])

def scrape_draft_from_game(html_content: str) -> pd.DataFrame:
    """
    Scrapes the draft from a game page.

    Args:
        html_content (str): The HTML content of the game page.

    Returns:
        pd.DataFrame: A DataFrame containing the draft.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    section_divs = soup.find_all("div", class_="col-2")
    bans = []
    picks = []
    section_name = ["Bans", "Picks"]
    for div in section_divs:
        if div.text.strip() == section_name[0]:
            champions_div = div.find_next_sibling("div", class_="col-10")
            bans.extend(
                img["alt"] for img in champions_div.find_all("img", class_="champion_icon_medium")
            )
        elif div.text.strip() == section_name[1]:
            champions_div = div.find_next_sibling("div", class_="col-10")
            picks.extend(
                img["alt"] for img in champions_div.find_all("img", class_="champion_icon_medium")
            )

    # Clean data by organizing picks and bans by team
    blue_bans = bans[:len(bans)//2]
    red_bans = bans[len(bans)//2:]
    blue_picks = picks[:len(picks)//2]
    red_picks = picks[len(picks)//2:]

    return pd.DataFrame({"blue_bans": blue_bans, "red_bans": red_bans, "blue_picks": blue_picks, "red_picks": red_picks})

def collect_match_patch(html_content: str) -> str:
    """
    Scrapes the patch from a match page.

    Args:
        html_content (str): The HTML content of the match page.

    Returns:
        str: The patch.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    patch_div = soup.find('div', class_='col-3 text-right')
    if patch_div:
        return patch_div.text.strip()[1:]
    return None

def collect_roster_from_match(html_content: str) -> pd.DataFrame:
    """
    Scrapes the roster from a match page.

    Args:
        html_content (str): The HTML content of the match page.

    Returns:
        pd.DataFrame: A DataFrame containing the roster.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    player_links = soup.find_all('a', class_='link-blanc')
    player_names = [link.get_text(strip=True) for link in player_links]
    df = pd.DataFrame()
    df['Blue Side Roster'] = player_names[:len(player_names)//2]
    df['Red Side Roster'] = player_names[len(player_names)//2:]

    return df
        

def scrape_full_tournament(tourney_url_endpoint: str):
    """
    Scrapes the full tournament data from a tournament page.

    Args:
        tourney_url_endpoint (str): The URL endpoint of the tournament.

    Returns:
        pd.DataFrame: A DataFrame containing the full tournament data.
    """
    upd_url = GOLGG_BASE_URL + GOLGG_TOURNAMENT_SERIES_ENDPOINT + tourney_url_endpoint
    try:
        res = requests.get(upd_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []

    df = pd.DataFrame()

    ln = get_all_games_from_tournament(res.text)
    for link in ln:
        try:
            game_response = requests.get(link, headers=HEADERS, timeout=15)
            game_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            continue
        game_html = game_response.text
        series_games = collect_matches_from_game(game_html)
        new_data = scrape_draft_from_game(game_html)
        new_data["patch"] = collect_match_patch(game_html)
        rosters = collect_roster_from_match(game_html)
        new_data["blue_roster"] = rosters["Blue Side Roster"]
        new_data["red_roster"] = rosters["Red Side Roster"]
        new_data = pd.concat([new_data, scrape_teams_side_winner_from_game(game_html)], ignore_index=True)
        df = pd.concat([df, new_data], ignore_index=True)

        for link_x in series_games:
            try:
                game_response = requests.get(link_x, headers=HEADERS, timeout=15)
                game_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                continue
            game_html = game_response.text
            new_data = scrape_draft_from_game(game_html)
            new_data["patch"] = collect_match_patch(game_html)
            rosters = collect_roster_from_match(game_html)
            new_data["blue_roster"] = rosters["Blue Side Roster"]
            new_data["red_roster"] = rosters["Red Side Roster"]
            new_data = pd.concat([new_data, scrape_teams_side_winner_from_game(game_html)], ignore_index=True)
            df = pd.concat([df, new_data], ignore_index=True)
    return df

def full_season_tourney_list(season: int) -> list[str]:
    """
    Scrapes the list of tournaments for a given season.

    Args:
        season (int): The season.

    Returns:
        list[str]: A list of tournament names.
    """
    try:
        res = requests.post(GOLGG_PHP_URL, headers=HEADER_DETAILED, data=GOL_GG_PAYLOAD_GEN(season), timeout=15)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []

    tourneys = res.json()

    tourney_names = [tourney['trname'] for tourney in tourneys]
    return tourney_names

def main():
    """
    Main function to run the script.
    """

    # for season in range(10, 12):
    #     for split in [Split.SPRING, Split.SUMMER, Split.WINTER]:
    #         c_url = GOLGG_BASE_URL + GOL_GG_PICKBAN_BY_PATCH_ENDPOINT + GOL_GG_SEASON_SPLIT_URL_GEN(season, split)
    #         df = scrape_pick_ban_by_patch(c_url)
    #         print(df)
    #         df.to_csv(f"pick_ban_by_patch_s{season}{split.name.lower().capitalize()}.csv", index=False)

    # tourney = "First Stand 2025/"
    # df = scrape_full_tournament(f"{GOLGG_BASE_URL}{GOLGG_TOURNAMENT_SERIES_ENDPOINT}{tourney}")
    # print(df)
    # df.to_csv(f"drafts_s15Spring.csv", index=False)

    # tourney = "First Stand 2025/"
    # try:
    #     res = requests.get(f"{GOLGG_BASE_URL}{GOLGG_TOURNAMENT_SERIES_ENDPOINT}{tourney}", headers=HEADERS)
    #     res.raise_for_status()
    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")
    #     return

    # print(get_all_games_from_tournament(res.text))

    # df = scrape_full_tournament("First Stand 2025/")

    # df.to_csv(f"drafts_FirstStand2025.csv", index=False)
    # print(df)

    # for season in range(14, 15):
    #     data = full_season_tourney_list(season)
    #     with open(f"tournaments_s{season}.txt", "w") as f:
    #         for tourney in data:
    #             f.write(f"{tourney}\n")


    # TODO: DO NOT DO NOT DO NOT TURN THIS ON UNLESS ABSOLUTELY NECESSARY
    for s_num in range(10, 15):
        with open (f"tournaments_by_season/tournaments_s{s_num}.txt", "r") as f:
            tourneys = f.readlines()

        for tourney in tourneys:
            df = scrape_full_tournament(f"{tourney.strip()}/")
            df.to_csv(f"tournament_draft_csvs/drafts_s{s_num}_{tourney.strip()}.csv", index=False)
    # ff = scrape_full_tournament("LCS Spring 2020/")
    # print(ff)


if __name__ == "__main__":
    main()
