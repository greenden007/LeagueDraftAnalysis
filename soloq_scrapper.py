"""
Riot API Scraper
Idan TODO refresh method (think, consult Rohan)
"""

import os
import time
import logging
import socket
import sys
import json
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Set

# ========================
# Configuration
# ========================
class Config:

    # Only Idan has the API key saved locally

    """Centralized configuration with validation"""
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL = "https://{region}.api.riotgames.com/lol"
    DATA_DRAGON_URL = "https://ddragon.leagueoflegends.com/realms/{region}.json"

    REGIONS = [
        # Americas
        "na1",   # North America
        "br1",   # Brazil
        "la1",   # Latin America North
        "la2",   # Latin America South

        # Europe
        "euw1",  # Europe West
        "eun1",  # Europe Nordic & East
        "tr1",   # Turkey

        # Asia
        "kr",    # South Korea
        "jp1",   # Japan
        
        # SEA
        "oc1",   # Oceania
        "ph2",   # Philippines
        "sg2",   # Singapore
        "th2",   # Thailand
        "tw2",   # Taiwan
        "vn2"    # Vietnam
    ]

    MATCH_REGION_MAP = {
        region: 
            "americas" if region in ["na1", "br1", "la1", "la2"] else
            "europe" if region in ["euw1", "eun1", "tr1"] else
            "asia" if region in ["kr", "jp1"] else
            "sea"  
        for region in REGIONS
    }

    CHAMPION_MAPPING = {
        "aatrox": "Aatrox", "ahri": "Ahri", "akali": "Akali", "akshan": "Akshan",
        "alistar": "Alistar", "amumu": "Amumu", "anivia": "Anivia", "annie": "Annie",
        "aphelios": "Aphelios", "ashe": "Ashe", "aurelionsol": "Aurelion Sol",
        "azir": "Azir", "bard": "Bard", "belveth": "Belveth", "blitzcrank": "Blitzcrank",
        "brand": "Brand", "braum": "Braum", "briar": "Briar", "caitlyn": "Caitlyn",
        "camille": "Camille", "cassiopeia": "Cassiopeia", "chogath": "Chogath",
        "corki": "Corki", "darius": "Darius", "diana": "Diana", "drmundo": "Dr. Mundo",
        "draven": "Draven", "ekko": "Ekko", "elise": "Elise", "evelynn": "Evelynn",
        "ezreal": "Ezreal", "fiddlesticks": "Fiddlesticks", "fiora": "Fiora",
        "fizz": "Fizz", "galio": "Galio", "gangplank": "Gangplank", "garen": "Garen",
        "gnar": "Gnar", "gragas": "Gragas", "graves": "Graves", "gwen": "Gwen",
        "hecarim": "Hecarim", "heimerdinger": "Heimerdinger", "hwei": "Hwei",
        "illaoi": "Illaoi", "irelia": "Irelia", "ivern": "Ivern", "janna": "Janna",
        "jarvaniv": "Jarvan IV", "jax": "Jax", "jayce": "Jayce", "jhin": "Jhin",
        "jinx": "Jinx", "kaisa": "Kaisa", "kalista": "Kalista", "karma": "Karma",
        "karthus": "Karthus", "kassadin": "Kassadin", "katarina": "Katarina",
        "kayle": "Kayle", "kayn": "Kayn", "kennen": "Kennen", "khazix": "KhaZix",
        "kindred": "Kindred", "kled": "Kled", "kogmaw": "KogMaw", "ksante": "KSante",
        "leblanc": "LeBlanc", "leesin": "Lee Sin", "leona": "Leona", "lillia": "Lillia",
        "lissandra": "Lissandra", "lucian": "Lucian", "lulu": "Lulu", "lux": "Lux",
        "malphite": "Malphite", "malzahar": "Malzahar", "maokai": "Maokai",
        "masteryi": "Master Yi", "milio": "Milio", "missfortune": "Miss Fortune",
        "mordekaiser": "Mordekaiser", "morgana": "Morgana", "naafiri": "Naafiri",
        "nami": "Nami", "nasus": "Nasus", "nautilus": "Nautilus", "neeko": "Neeko",
        "nidalee": "Nidalee", "nilah": "Nilah", "nocturne": "Nocturne",
        "nunu": "Nunu & Willump", "olaf": "Olaf", "orianna": "Orianna", "ornn": "Ornn",
        "pantheon": "Pantheon", "poppy": "Poppy", "pyke": "Pyke", "qiyana": "Qiyana",
        "quinn": "Quinn", "rakan": "Rakan", "rammus": "Rammus", "reksai": "Reksai",
        "rell": "Rell", "renataglasc": "Renata Glasc", "renekton": "Renekton",
        "rengar": "Rengar", "riven": "Riven", "rumble": "Rumble", "ryze": "Ryze",
        "samira": "Samira", "sejuani": "Sejuani", "senna": "Senna",
        "seraphine": "Seraphine", "sett": "Sett", "shaco": "Shaco", "shen": "Shen",
        "shyvana": "Shyvana", "singed": "Singed", "sion": "Sion", "sivir": "Sivir",
        "skarner": "Skarner", "sona": "Sona", "soraka": "Soraka", "swain": "Swain",
        "sylas": "Sylas", "syndra": "Syndra", "tahmkench": "Tahm Kench",
        "taliyah": "Taliyah", "talon": "Talon", "taric": "Taric", "teemo": "Teemo",
        "thresh": "Thresh", "tristana": "Tristana", "trundle": "Trundle",
        "tryndamere": "Tryndamere", "twistedfate": "Twisted Fate", "twitch": "Twitch",
        "udyr": "Udyr", "urgot": "Urgot", "varus": "Varus", "vayne": "Vayne",
        "veigar": "Veigar", "velkoz": "VelKoz", "vex": "Vex", "vi": "Vi",
        "viego": "Viego", "viktor": "Viktor", "vladimir": "Vladimir",
        "volibear": "Volibear", "warwick": "Warwick", "monkeyking": "Wukong",
        "xayah": "Xayah", "xerath": "Xerath", "xinzhao": "Xin Zhao", "yasuo": "Yasuo",
        "yone": "Yone", "yorick": "Yorick", "yuumi": "Yuumi", "zac": "Zac",
        "zed": "Zed", "zeri": "Zeri", "ziggs": "Ziggs", "zilean": "Zilean",
        "zoe": "Zoe", "zyra": "Zyra"
    }

    RATE_LIMITS = {
        "app": {"per_two_minutes": 95, "window_seconds": 120},
        "endpoints": {
            "league": {"per_two_minutes": 85, "window_seconds": 120},
            "match": {"per_two_minutes": 85, "window_seconds": 120},
            "summoner": {"per_two_minutes": 85, "window_seconds": 120}
        }
    }

    MAX_MATCHES_PER_PLAYER = 100
    MIN_GAMES_THRESHOLD = 1
    BASE_OUTPUT_DIR = "soloq_stats"
    PATCH_VERSIONS_TO_KEEP = 5
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 3]
    SYNERGY_COMBOS = [
        "jungle_mid", "jungle_adc", "support_adc", "top_jungle",
        "support_jungle", "jungle_support_mid", "jungle_top_mid",
        "jungle_adc_support"
    ]

    @classmethod
    def normalize_champion_name(cls, name: str) -> str:
        return cls.CHAMPION_MAPPING.get(name.lower(), name.title())
    
    @classmethod
    def get_role_name(cls, position: str) -> str:
        role_map = {
            "TOP": "Top",
            "JUNGLE": "Jungle",
            "MIDDLE": "Mid",
            "BOTTOM": "ADC",
            "UTILITY": "Support"
        }
        return role_map.get(position, position)
    
    @classmethod
    def get_current_patch(cls) -> str:
        try:
            response = requests.get(cls.DATA_DRAGON_URL.format(region="na"), timeout=5)
            response.raise_for_status()
            full_version = response.json()['v']
            year, patch_num = full_version.split('.')[:2]
            return f"{year}.{patch_num}"
        except Exception:
            current_year = datetime.now().year % 100
            return f"{current_year}.9"

    @classmethod
    def validate(cls, api: 'RiotAPI'):
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")
        os.makedirs(cls.BASE_OUTPUT_DIR, exist_ok=True)

# ========================
# Data Models
# ========================
class ChampionStats:
    def __init__(self):
        self.roles = defaultdict(lambda: {
            "games": 0, "wins": 0, "kills": 0, "deaths": 0, "assists": 0
        })
        self.matchups = defaultdict(lambda: defaultdict(lambda: {
            "games": 0, "wins": 0, "kills": 0, "deaths": 0, "assists": 0
        }))

    def add_game(self, win: bool, kills: int, deaths: int, assists: int, 
                 role: str, opponent: Optional[str] = None):
        
        role_stats = self.roles[role]
        role_stats["games"] += 1
        role_stats["wins"] += int(win)
        role_stats["kills"] += kills
        role_stats["deaths"] += deaths
        role_stats["assists"] += assists
        
        if opponent:
            opponent = Config.normalize_champion_name(opponent)
            matchup = self.matchups[role][opponent]
            matchup["games"] += 1
            matchup["wins"] += int(win)
            matchup["kills"] += kills
            matchup["deaths"] += deaths
            matchup["assists"] += assists

class SynergyStats:
    def __init__(self):
        self.games = 0
        self.wins = 0
        
    def add_game(self, win: bool):
        self.games += 1
        self.wins += int(win)


# ========================
# Rate Limiter
# ========================
class PrecisionRateLimiter:
    def __init__(self):
        self.request_times = {
            "app": deque(maxlen=Config.RATE_LIMITS["app"]["per_two_minutes"]),
            "league": deque(maxlen=Config.RATE_LIMITS["endpoints"]["league"]["per_two_minutes"]),
            "match": deque(maxlen=Config.RATE_LIMITS["endpoints"]["match"]["per_two_minutes"]),
            "summoner": deque(maxlen=Config.RATE_LIMITS["endpoints"]["summoner"]["per_two_minutes"])
        }
        self.last_request_time = 0
        self.total_requests = 0

    def wait(self, endpoint_type: str) -> None:
        now = time.time()
        self.total_requests += 1
        delays = []
        for limit_type in ["app", endpoint_type]:
            queue = self.request_times[limit_type]
            max_requests = Config.RATE_LIMITS["app"]["per_two_minutes"] if limit_type == "app" else Config.RATE_LIMITS["endpoints"][endpoint_type]["per_two_minutes"]
            window = Config.RATE_LIMITS["app"]["window_seconds"] if limit_type == "app" else Config.RATE_LIMITS["endpoints"][endpoint_type]["window_seconds"]
            
            if len(queue) >= max_requests:
                elapsed = now - queue[0]
                if elapsed < window:
                    delays.append((queue[0] + window) - now)
        
        if delays:
            time.sleep(max(delays))
        
        current_time = time.time()
        self.request_times["app"].append(current_time)
        self.request_times[endpoint_type].append(current_time)
        self.last_request_time = current_time

# ========================
# API Client
# ========================
class RiotAPI:
    """Riot API client with improved connectivity logging"""
    
    def __init__(self, rate_limiter: PrecisionRateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": Config.API_KEY,
            "Accept": "application/json",
            "User-Agent": "LeagueDataCollector/10.0 (GlobalStats)"
        })
        self.total_requests = 0
        self.failed_requests = 0
        self.connectivity_checked = False
        self.last_connection_state = True

    def check_connectivity(self) -> None:
        """Monitor internet connection with state-aware logging"""
        while True:
            try:
                socket.create_connection(("1.1.1.1", 53), timeout=5)
                
                if not self.last_connection_state:
                    logger.info("🌐✅ Internet connection restored! Resuming data collection...")
                    self.last_connection_state = True
                
                self.connectivity_checked = True
                return
            
            except OSError:
                if self.last_connection_state:
                    logger.warning("🌐❌ Internet connection lost! Pausing until restored...")
                    self.last_connection_state = False
                else:
                    logger.info("🌐⏳ Still waiting for internet connection...")
                
                time.sleep(10)

    def get_json(self, url: str, endpoint_type: str) -> Optional[Dict]:
        """Make API request with connection-aware error handling"""
        logger.debug(f"🌐 Sending {endpoint_type} request to: {url}")
        if not self.connectivity_checked:
            self.check_connectivity()

        for attempt in range(Config.MAX_RETRIES):
            self.rate_limiter.wait(endpoint_type)
            
            try:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                self.total_requests += 1
                
                if response.status_code == 404:
                    logger.debug(f"🔍 Resource not found: {url}")
                    return None
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"⏳🔁 Rate limited on {endpoint_type}. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                self.failed_requests += 1
                logger.debug(f"🌐🔌 Connection error: {str(e)}")
                self.check_connectivity() 
                continue
            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                logger.debug(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
                time.sleep(Config.RETRY_DELAYS[attempt])
                
        logger.warning(f"⚠️ Failed after {Config.MAX_RETRIES} attempts for {url}")
        return None

    def get_challenger_league(self, region: str) -> Optional[Dict]:
        url = f"{Config.BASE_URL.format(region=region)}/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self.get_json(url, "league")

    def get_match_history(self, puuid: str, region: str) -> Optional[List[str]]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={Config.MAX_MATCHES_PER_PLAYER}"
        return self.get_json(url, "match")

    def get_match_details(self, match_id: str, region: str) -> Optional[Dict]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self.get_json(url, "match")


# ========================
# Data Storage Manager
# ========================
class DataStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.global_stats = defaultdict(ChampionStats)
        self.synergies = {combo: defaultdict(SynergyStats) for combo in Config.SYNERGY_COMBOS}
        self.load_existing_data()
        
    
    def load_existing_data(self):
        global_dir = os.path.join(self.base_dir, "global_stats")
        if os.path.exists(global_dir):
            for file in os.listdir(global_dir):
                if file.endswith('.csv'):
                    role = file.replace('.csv', '')
                    df = pd.read_csv(os.path.join(global_dir, file))
                    for _, row in df.iterrows():
                        champ = row['champion']
                        stats = self.global_stats[champ]
                        stats.roles[role] = {
                            "games": row['games'],
                            "wins": row['wins'],
                            "kills": row['kills'],
                            "deaths": row['deaths'],
                            "assists": row['assists']
                        }
        
        matchup_dir = os.path.join(self.base_dir, "matchups")
        if os.path.exists(matchup_dir):
            for champ_folder in os.listdir(matchup_dir):
                champ_dir = os.path.join(matchup_dir, champ_folder)
                if os.path.isdir(champ_dir):
                    for file in os.listdir(champ_dir):
                        if file.endswith('.csv'):
                            role = file.replace('.csv', '')
                            df = pd.read_csv(os.path.join(champ_dir, file))
                            for _, row in df.iterrows():
                                opponent = row['opponent']
                                self.global_stats[champ_folder].matchups[role][opponent] = {
                                    "games": row['games'],
                                    "wins": row['wins'],
                                    "kills": row['kills'],
                                    "deaths": row['deaths'],
                                    "assists": row['assists']
                                }
        
        synergy_dir = os.path.join(self.base_dir, "synergies")
        if os.path.exists(synergy_dir):
            for file in os.listdir(synergy_dir):
                if file.endswith('.csv'):
                    combo = file.replace('.csv', '')
                    df = pd.read_csv(os.path.join(synergy_dir, file))
                    for _, row in df.iterrows():
                        if combo in ("jungle_mid", "jungle_adc", "support_adc", 
                                   "top_jungle", "support_jungle"):
                            key = (row['champion1'], row['champion2'])
                        else:
                            key = (row['champion1'], row['champion2'], row['champion3'])
                        stats = SynergyStats()
                        stats.games = row['games']
                        stats.wins = row['wins']
                        self.synergies[combo][key] = stats
    
    def save_data(self):
        
        global_dir = os.path.join(self.base_dir, "global_stats")
        matchup_dir = os.path.join(self.base_dir, "matchups")
        synergy_dir = os.path.join(self.base_dir, "synergies")
        os.makedirs(global_dir, exist_ok=True)
        os.makedirs(matchup_dir, exist_ok=True)
        os.makedirs(synergy_dir, exist_ok=True)

        for role in ["Top", "Jungle", "Mid", "ADC", "Support"]:
            role_stats = []
            for champ, stats in self.global_stats.items():
                role_data = stats.roles.get(role, {})
                if role_data.get("games", 0) > 0:
                    games = role_data["games"]
                    wins = role_data["wins"]
                    role_stats.append({
                        "champion": champ,
                        "games": games,
                        "wins": wins,
                        "kills": role_data["kills"],
                        "deaths": role_data["deaths"],
                        "assists": role_data["assists"],
                        "win_rate": round((wins / games) * 100, 2) if games > 0 else 0,
                        "kda": round((role_data["kills"] + role_data["assists"]) / max(1, role_data["deaths"]), 2),
                        "avg_kills": round(role_data["kills"] / games, 2),
                        "avg_deaths": round(role_data["deaths"] / games, 2),
                        "avg_assists": round(role_data["assists"] / games, 2)
                    })
            
            if role_stats:
                pd.DataFrame(role_stats).to_csv(
                    os.path.join(global_dir, f"{role}.csv"), 
                    index=False
                )
        
        for champ, stats in self.global_stats.items():
            champ_dir = os.path.join(matchup_dir, champ)
            os.makedirs(champ_dir, exist_ok=True)
            
            for role, matchups in stats.matchups.items():
                matchup_data = []
                for opponent, data in matchups.items():
                    if data["games"] > 0:
                        matchup_data.append({
                            "opponent": opponent,
                            "games": data["games"],
                            "wins": data["wins"],
                            "kills": data["kills"],
                            "deaths": data["deaths"],
                            "assists": data["assists"],
                            "win_rate": round((data["wins"] / data["games"]) * 100, 2),
                            "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                            "avg_kills": round(data["kills"] / data["games"], 2),
                            "avg_deaths": round(data["deaths"] / data["games"], 2),
                            "avg_assists": round(data["assists"] / data["games"], 2)
                        })
                
                if matchup_data:
                    pd.DataFrame(matchup_data).to_csv(
                        os.path.join(champ_dir, f"{role}.csv"), 
                        index=False
                    )
        
        for combo, stats_dict in self.synergies.items():
            synergies = []
            for champs, stats in stats_dict.items():
                if stats.games > 0:
                    if len(champs) == 2:
                        synergies.append({
                            "champion1": champs[0],
                            "champion2": champs[1],
                            "games": stats.games,
                            "wins": stats.wins,
                            "win_rate": round((stats.wins / stats.games) * 100, 2)
                        })
                    else:
                        synergies.append({
                            "champion1": champs[0],
                            "champion2": champs[1],
                            "champion3": champs[2],
                            "games": stats.games,
                            "wins": stats.wins,
                            "win_rate": round((stats.wins / stats.games) * 100, 2)
                        })
            if synergies:
                pd.DataFrame(synergies).to_csv(
                    os.path.join(synergy_dir, f"{combo}.csv"), 
                    index=False
                )
        
        champion_roles = {}
        for champ, stats in self.global_stats.items():
            if any(role_data.get("games", 0) > 0 for role_data in stats.roles.values()):
                roles_played = [role for role, role_data in stats.roles.items() 
                                if role_data.get("games", 0) > 0]
                if roles_played:
                    champion_roles[champ] = sorted(roles_played)
        
        if champion_roles:
            with open("champion_roles.json", "w") as f:
                json.dump(champion_roles, f, indent=2)



# ========================
# Patch Tracker
# ========================
class PatchTracker:
    def __init__(self):
        self.file_path = ""
        self.target_patches = []
        self.scraped_counts = defaultdict(int)
        self.processed_match_ids = defaultdict(set)
        self.all_processed_match_ids = set()
        
    
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                data = json.load(f)
                self.target_patches = data.get("target_patches", [])
                self.scraped_counts = defaultdict(int, data.get("scraped_counts", {}))
                self.processed_match_ids = defaultdict(set, {
                    patch: set(match_ids) 
                    for patch, match_ids in data.get("processed_match_ids", {}).items()
                })
                self.all_processed_match_ids = set()
                for match_ids_set in self.processed_match_ids.values():
                    self.all_processed_match_ids |= match_ids_set
    
    def save(self):
        data = {
            "target_patches": self.target_patches,
            "scraped_counts": dict(self.scraped_counts),
            "processed_match_ids": {
                patch: list(match_ids) 
                for patch, match_ids in self.processed_match_ids.items()
            }
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def update_target_patches(self, current_patch: str) -> List[str]:
        try:
            year_str, patch_num_str = current_patch.split('.')
            current_year = int(year_str)
            current_patch_num = int(patch_num_str)
        except ValueError:
            logger.error(f"Invalid current patch format: {current_patch}")
            return []
        
        patches = []
        for i in range(4, -1, -1):
            patch_number = current_patch_num - i
            if patch_number > 0:
                patches.append(f"{current_year}.{patch_number}")
        
        if len(patches) < 5:
            remaining = 5 - len(patches)
            previous_year = current_year - 1
            for patch_number in range(12, 12 - remaining, -1):
                if patch_number > 0:
                    patches.insert(0, f"{previous_year}.{patch_number}")
        
        self.target_patches = patches[-5:]
        
        for patch in list(self.scraped_counts.keys()):
            if patch not in self.target_patches:
                del self.scraped_counts[patch]
        for patch in list(self.processed_match_ids.keys()):
            if patch not in self.target_patches:
                del self.processed_match_ids[patch]
        
        for patch in self.target_patches:
            if patch not in self.scraped_counts:
                self.scraped_counts[patch] = 0
            if patch not in self.processed_match_ids:
                self.processed_match_ids[patch] = set()
        
        self.all_processed_match_ids = set()
        for match_ids_set in self.processed_match_ids.values():
            self.all_processed_match_ids |= match_ids_set
        
        return self.target_patches

    def add_processed_match(self, match_id: str, patch: str):
        if match_id in self.all_processed_match_ids:
            logger.warning(f"⚠️ Attempted to add duplicate match: {match_id}")
            return
            
        self.processed_match_ids[patch].add(match_id)
        self.all_processed_match_ids.add(match_id)
        self.scraped_counts[patch] += 1
        logger.debug(f"➕ Added match {match_id[:8]} for patch {patch}")


# ========================
# Main Scraper
# ========================
class LeagueScraper:
    def __init__(self):
        self.rate_limiter = PrecisionRateLimiter()
        self.api = RiotAPI(self.rate_limiter)
        self.current_patch = Config.get_current_patch()
        self.patch_tracker = PatchTracker()
        self.target_patches = self.patch_tracker.update_target_patches(self.current_patch)
        self.patch_range = self.get_patch_range_name()
        self.base_dir = os.path.join(Config.BASE_OUTPUT_DIR, self.patch_range)
        self.patch_tracker.file_path = os.path.join(self.base_dir, "patch_tracker.json")
        self.patch_tracker.load()
        self.data_store = DataStore(self.base_dir)
        self.start_time = time.time()
        self.processed_players = 0
        self.skipped_players = 0
        self.failed_summoner_lookups = 0
        self.processed_match_ids = set()
        self.processed_players_for_match = defaultdict(set)
        self.new_matches_this_run = 0
        
        if os.path.exists(self.base_dir):
            logger.info(f"📁 Data for patch range {self.patch_range} already exists. Exiting.")
            sys.exit(0)
            
        os.makedirs(self.base_dir, exist_ok=True)
        
        logger.info(f"🎯 Target patches: {', '.join(self.target_patches)}")

    
    def get_patch_range_name(self) -> str:
        """Generate folder name from patch range (min_patch-max_patch)"""
        if not self.target_patches:
            return "unknown_patches"
        sorted_patches = sorted(
            self.target_patches,
            key=lambda x: tuple(map(int, x.split('.'))))
        return f"{sorted_patches[0]}-{sorted_patches[-1]}"

    def run(self) -> None:
        try:
            for region in Config.REGIONS:
                logger.info(f"🌍 Starting {region.upper()} processing")
                self.process_region(region)
                logger.info(f"✅ Finished {region.upper()} processing")
            
            self.data_store.save_data()
            self.patch_tracker.save()
            self.update_champion_roles()
            self.log_final_stats()
        except KeyboardInterrupt:
            logger.info("🛑 Manual interrupt received")
            self.data_store.save_data()
            self.patch_tracker.save()
            self.log_final_stats()
            sys.exit(0)
        except Exception as e:
            logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
            self.data_store.save_data()
            self.patch_tracker.save()
            raise


    def is_valid_match(self, match: Dict, puuid: str) -> bool:
        try:
            return any(p["puuid"] == puuid for p in match["info"]["participants"])
        except KeyError:
            return False

    def process_region(self, region: str) -> None:
        ladder = self.api.get_challenger_league(region)
        if not ladder or "entries" not in ladder:
            logger.error(f"❌ Invalid ladder data for {region}, skipping region")
            return

        players = ladder["entries"]
        logger.info(f"👥 Found {len(players)} players in {region.upper()}")
        
        for i, player in enumerate(players):
            if "puuid" not in player:
                logger.warning(f"⚠️ Player entry {i+1} missing puuid, skipping")
                self.skipped_players += 1
                continue
                
            puuid = player["puuid"]
            logger.info(f"👤 Processing player {i+1}/{len(players)}")
            result = self.process_player(region, puuid)
            
            if result[0]:
                logger.info(f"✅ Success: {result[1]}")
                self.processed_players += 1
            else:
                logger.warning(f"⚠️ Skip: {result[1]}")
                self.skipped_players += 1
                if "summoner not found" in result[1].lower():
                    self.failed_summoner_lookups += 1
    
    def process_player(self, region: str, puuid: str) -> Tuple[bool, str]: # Rohan take a look here
        match_ids = self.api.get_match_history(puuid, region)
        if not match_ids:
            return (False, f"No matches for {puuid[:6]}")

        valid_matches = []
        new_entries = 0
        duplicate_entries = 0
        invalid_patch = 0
        new_matches = 0
        
        for match_id in match_ids[:Config.MAX_MATCHES_PER_PLAYER]:
            if match_id in self.processed_players_for_match and puuid in self.processed_players_for_match[match_id]:
                duplicate_entries += 1
                continue
                
            match = self.api.get_match_details(match_id, region)
            if not match:
                continue
            
            if "info" not in match or "gameVersion" not in match["info"]:
                logger.warning(f"⚠️ Match {match_id} missing version info")
                continue
                
            patch = self.get_match_patch(match)
            if patch not in self.target_patches:
                invalid_patch += 1
                continue
                
            valid_matches.append(match)
            if match_id not in self.processed_players_for_match:
                self.processed_players_for_match[match_id] = set()
            self.processed_players_for_match[match_id].add(puuid)
            new_entries += 1

            if match_id not in self.patch_tracker.all_processed_match_ids:
                new_matches += 1
                self.new_matches_this_run += 1
                self.patch_tracker.add_processed_match(match_id, patch)

        logger.debug(f"🔍 Player matches: {len(match_ids)} total, "
                    f"{new_entries} new entries, {duplicate_entries} duplicate entries, "
                    f"{invalid_patch} wrong patch")
        
        if not valid_matches:
            return (False, f"No valid matches for {puuid[:6]}")
            
        logger.info(f"📊 Processing {len(valid_matches)} matches "
                    f"({new_entries} new player entries, {new_matches} new matches)")
        for match in valid_matches:
            self.process_match(match)
            
        return (True, f"Processed {len(valid_matches)} matches "
                    f"({new_entries} new player entries, {new_matches} new matches)")

    def get_match_patch(self, match: Dict) -> str:
        version_str = match["info"]["gameVersion"]
        parts = version_str.split('.')
        
        if len(parts) < 2:
            logger.warning(f"⚠️ Unexpected version format: {version_str}")
            return "0.0"
        
        return f"{parts[0]}.{parts[1]}"

    def process_match(self, match: Dict) -> None:
        try:
            participants = match["info"]["participants"]
            teams = defaultdict(list)
            
            for p in participants:
                team_id = p["teamId"]
                role = Config.get_role_name(p["teamPosition"])
                champ = Config.normalize_champion_name(p["championName"])
                
                teams[team_id].append((role, champ, p))
                
                stats = self.data_store.global_stats[champ]
                stats.add_game(
                    win=p["win"],
                    kills=p["kills"],
                    deaths=p["deaths"],
                    assists=p["assists"],
                    role=role
                )
            
            for p in participants:
                role = Config.get_role_name(p["teamPosition"])
                if role not in ("Top", "Mid", "ADC", "Support"):
                    continue
                
                champ = Config.normalize_champion_name(p["championName"])
                opponent = self.get_lane_opponent(p, participants)
                
                if opponent:
                    opp_champ = Config.normalize_champion_name(opponent["championName"])
                    
                    self.data_store.global_stats[champ].add_game(
                        win=p["win"],
                        kills=p["kills"],
                        deaths=p["deaths"],
                        assists=p["assists"],
                        role=role,
                        opponent=opp_champ
                    )
                    self.data_store.global_stats[opp_champ].add_game(
                        win=opponent["win"],
                        kills=opponent["kills"],
                        deaths=opponent["deaths"],
                        assists=opponent["assists"],
                        role=role,
                        opponent=champ
                    )
            
            # Process synergies
            for team_id, team_players in teams.items():
                role_champs = {role: champ for role, champ, _ in team_players}
                team_win = any(p["win"] for _, _, p in team_players)
                
                # Jungle + Mid
                if "Jungle" in role_champs and "Mid" in role_champs:
                    key = (role_champs["Jungle"], role_champs["Mid"])
                    self.data_store.synergies["jungle_mid"][key].add_game(team_win)
                
                # Jungle + ADC
                if "Jungle" in role_champs and "ADC" in role_champs:
                    key = (role_champs["Jungle"], role_champs["ADC"])
                    self.data_store.synergies["jungle_adc"][key].add_game(team_win)
                
                # Support + ADC
                if "Support" in role_champs and "ADC" in role_champs:
                    key = (role_champs["Support"], role_champs["ADC"])
                    self.data_store.synergies["support_adc"][key].add_game(team_win)
                
                # Top + Jungle
                if "Top" in role_champs and "Jungle" in role_champs:
                    key = (role_champs["Top"], role_champs["Jungle"])
                    self.data_store.synergies["top_jungle"][key].add_game(team_win)
                
                # Support + Jungle
                if "Support" in role_champs and "Jungle" in role_champs:
                    key = (role_champs["Support"], role_champs["Jungle"])
                    self.data_store.synergies["support_jungle"][key].add_game(team_win)
                
                # Jungle + Support + Mid
                if "Jungle" in role_champs and "Support" in role_champs and "Mid" in role_champs:
                    key = (
                        role_champs["Jungle"], 
                        role_champs["Support"], 
                        role_champs["Mid"]
                    )
                    self.data_store.synergies["jungle_support_mid"][key].add_game(team_win)
                
                # Jungle + Top + Mid
                if "Jungle" in role_champs and "Top" in role_champs and "Mid" in role_champs:
                    key = (
                        role_champs["Jungle"], 
                        role_champs["Top"], 
                        role_champs["Mid"]
                    )
                    self.data_store.synergies["jungle_top_mid"][key].add_game(team_win)
                
                # Jungle + ADC + Support
                if "Jungle" in role_champs and "ADC" in role_champs and "Support" in role_champs:
                    key = (
                        role_champs["Jungle"], 
                        role_champs["ADC"], 
                        role_champs["Support"]
                    )
                    self.data_store.synergies["jungle_adc_support"][key].add_game(team_win)
                    
        except Exception as e:
            logger.error(f"Match processing error: {str(e)}")

    def get_lane_opponent(self, player: Dict, participants: List[Dict]) -> Optional[Dict]:
        try:
            position = player["teamPosition"]
            team_id = player["teamId"]
            return next(
                (p for p in participants 
                 if p["teamId"] != team_id and p.get("teamPosition") == position),
                None
            )
        except Exception:
            return None

    def log_final_stats(self) -> None:
        total_time = (time.time() - self.start_time) / 60
        logger.info("\n📊 Final Statistics:")
        logger.info(f"⏱️  Total runtime: {total_time:.1f} minutes")
        logger.info(f"✅ Players processed: {self.processed_players}")
        logger.info(f"⚠️  Players skipped: {self.skipped_players}")
        logger.info(f"🔍 Failed summoner lookups: {self.failed_summoner_lookups}")
        logger.info(f"📡 API requests: {self.api.total_requests}")
        logger.info(f"❌ Failed requests: {self.api.failed_requests}")
        
        logger.info("\n📦 Patch Statistics:")
        for patch in self.target_patches:
            count = self.patch_tracker.scraped_counts.get(patch, 0)
            logger.info(f"  - Patch {patch}: {count} matches (total)")
        logger.info(f"  + This run added: {self.new_matches_this_run} new matches")
        
        logger.info("🏁 Script completed")
        logger.info("🏁")
        logger.info("🏁")
        logger.info("🏁")
        logger.info("🏁")
        logger.info("🏁")

# ========================
# Logging Setup
# ========================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logging.addLevelName(25, "SUCCESS")
    setattr(logger, 'success', lambda msg, *args: logger._log(25, msg, args))
    
    file = logging.FileHandler("scraper.log")
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file)
    
    return logger



# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    logger = setup_logging()
    
    rate_limiter = PrecisionRateLimiter()
    api = RiotAPI(rate_limiter)
    Config.validate(api=api)
    
    try:
        scraper = LeagueScraper()
        scraper.run()
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
        raise