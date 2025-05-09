"""
Riot API Scraper - Global Aggregated Stats with Matchups (Patch-Based)
"""

import os
import time
import logging
import socket
import sys
import re
import shutil
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

    @classmethod
    def normalize_champion_name(cls, name: str) -> str:
        return cls.CHAMPION_MAPPING.get(name.lower(), name.title())

    @classmethod
    def get_current_patch(cls) -> str:
        """Get current patch version from Data Dragon with year-based format"""
        try:
            response = requests.get(cls.DATA_DRAGON_URL.format(region="na"), timeout=5)
            response.raise_for_status()
            full_version = response.json()['v']


            year, patch_num = full_version.split('.')[:2]
            return f"{year}.{patch_num}"  

        except Exception as e:
            logger.error(f"Failed to get current patch: {str(e)}")

            current_year = datetime.datetime.now().year % 100  
            return f"{current_year}.9"  

    @classmethod
    def validate(cls, api: 'RiotAPI'):
        """Validate configuration and setup directories"""
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")

        os.makedirs(cls.BASE_OUTPUT_DIR, exist_ok=True)
        cls.cleanup_old_patches()

    @classmethod
    def get_patch_output_dir(cls, patch: str) -> str:
        """Get output directory path for a specific patch"""
        patch_dir = os.path.join(cls.BASE_OUTPUT_DIR, f"patch_{patch}")
        os.makedirs(os.path.join(patch_dir, "matchups"), exist_ok=True)
        return patch_dir

    @classmethod
    def cleanup_old_patches(cls):
        """Remove data for patches older than PATCH_VERSIONS_TO_KEEP"""
        try:
            dirs = [d for d in os.listdir(cls.BASE_OUTPUT_DIR) 
                    if os.path.isdir(os.path.join(cls.BASE_OUTPUT_DIR, d))]

            patches = []
            for d in dirs:
                match = re.match(r'patch_(\d+\.\d+)', d)
                if match:
                    year, num = map(int, match.group(1).split('.'))
                    patches.append((year, num, d))

            patches.sort(reverse=True, key=lambda x: (x[0], x[1]))
            
            for patch in patches[cls.PATCH_VERSIONS_TO_KEEP:]:
                shutil.rmtree(os.path.join(cls.BASE_OUTPUT_DIR, patch[2]))
                logger.info(f"Cleaned up old patch: {patch[2]}")
                
        except Exception as e:
            logger.error(f"Patch cleanup failed: {str(e)}")

# ========================
# Data Models
# ========================
class ChampionStats:
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.matchups = defaultdict(lambda: {
            "games": 0, "wins": 0, "kills": 0, "deaths": 0, "assists": 0
        })

    def add_game(self, win: bool, kills: int, deaths: int, assists: int, opponent: Optional[str] = None):
        self.games += 1
        self.wins += int(win)
        self.kills += kills
        self.deaths += deaths
        self.assists += assists
        
        if opponent:
            opponent = Config.normalize_champion_name(opponent)
            self.matchups[opponent]["games"] += 1
            self.matchups[opponent]["wins"] += int(win)
            self.matchups[opponent]["kills"] += kills
            self.matchups[opponent]["deaths"] += deaths
            self.matchups[opponent]["assists"] += assists

    def to_dict(self) -> Dict:
        return {
            "games": self.games,
            "wins": self.wins,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "matchups": dict(self.matchups)
        }

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games) * 100 if self.games else 0

    @property
    def kda(self) -> float:
        return (self.kills + self.assists) / max(1, self.deaths)

    def get_matchup_stats(self) -> List[Dict]:
        return [{
            "opponent": opponent,
            "games": data["games"],
            "win_rate": round((data["wins"] / data["games"]) * 100, 2),
            "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
            "avg_kills": round(data["kills"] / data["games"], 2),
            "avg_deaths": round(data["deaths"] / data["games"], 2),
            "avg_assists": round(data["assists"] / data["games"], 2),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        } for opponent, data in self.matchups.items()]

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

    def get_summoner_by_id(self, region: str, summoner_id: str) -> Optional[Dict]:
        url = f"{Config.BASE_URL.format(region=region)}/summoner/v4/summoners/{summoner_id}"
        return self.get_json(url, "summoner")

    def get_match_history(self, puuid: str, region: str) -> Optional[List[str]]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count={Config.MAX_MATCHES_PER_PLAYER}"
        return self.get_json(url, "match")

    def get_match_details(self, match_id: str, region: str) -> Optional[Dict]:
        match_region = Config.MATCH_REGION_MAP[region]
        url = f"https://{match_region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self.get_json(url, "match")

# ========================
# Main Scraper
# ========================
class LeagueScraper:
    def __init__(self):
        self.rate_limiter = PrecisionRateLimiter()
        self.api = RiotAPI(self.rate_limiter)
        self.target_patches = self.get_target_patches()
        self.existing_patches = self.get_existing_patches()
        self.active_patches = [p for p in self.target_patches if p not in self.existing_patches]
        self.global_stats = defaultdict(lambda: defaultdict(ChampionStats))
        self.start_time = time.time()
        self.processed_players = 0
        self.skipped_players = 0
        self.failed_summoner_lookups = 0

        logger.info(f"📂 Existing patches: {', '.join(self.existing_patches) or 'None'}")
        logger.info(f"🎯 Target patches: {', '.join(self.target_patches)}")
        logger.info(f"🚀 Active patches to scrape: {', '.join(self.active_patches) or 'None'}")

    def get_target_patches(self) -> List[str]:
        """Get last 5 patches including current within the current year"""
        current = Config.get_current_patch()  
        try:
            year_str, patch_num_str = current.split('.')
            current_year = int(year_str) + 10
            current_patch_num = int(patch_num_str)
        except ValueError:
            logger.error(f"Invalid current patch format: {current}")
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
                patches.insert(0, f"{previous_year}.{patch_number}")

        return sorted(patches[-5:])  
    
    def get_existing_patches(self) -> Set[str]:
        """Get set of existing patch directories"""
        return {d.split('_')[1] for d in os.listdir(Config.BASE_OUTPUT_DIR)
                if os.path.isdir(os.path.join(Config.BASE_OUTPUT_DIR, d)) and d.startswith('patch_')}

    def run(self) -> None:
        if not self.active_patches:
            logger.info("✅ All target patches already exist - nothing to scrape")
            return

        try:
            # Process all regions first
            for region in Config.REGIONS:
                logger.info(f"🌍 Starting {region.upper()} processing")
                self.process_region(region)
                logger.info(f"✅ Finished {region.upper()} processing")
            
            self.save_data()
            time.sleep(1)
            self.global_stats.clear()  
            self.log_final_stats()

        except KeyboardInterrupt:
            logger.info("🛑 Manual interrupt received")
            if self._has_data():
                self.save_data()  
                time.sleep(1)
                self.global_stats.clear()
            self.log_final_stats()
            sys.exit(0)
        except Exception as e:
            logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
            if self._has_data():
                self.save_data()  
                time.sleep(1)
                self.global_stats.clear()
            raise 

    def _has_data(self) -> bool:
        """Check if any meaningful data exists"""
        return any(
            champ_stats.games > 0
            for patch_stats in self.global_stats.values()
            for champ_stats in patch_stats.values()
        )
    
    def is_valid_match(self, match: Dict, puuid: str) -> bool:
        try:
            return any(p["puuid"] == puuid for p in match["info"]["participants"])
        except KeyError:
            return False
        
    def process_region(self, region: str) -> None:
        logger.info(f"\n🌍🚀 Starting {region.upper()} region processing")
        ladder = self.api.get_challenger_league(region)
        if not ladder or "entries" not in ladder:
            logger.error(f"❌🚫 Invalid ladder data for {region}, skipping region")
            return

        players = ladder["entries"]
        logger.info(f"👥🔍 Found {len(players)} players in {region.upper()}")
        
        for i, player in enumerate(players):
            logger.info(f"👤 Processing player {i+1}/{len(players)}")
            logger.debug(f"📇 Summoner ID: {player['summonerId'][:6]}...")
            result = self.process_player(region, player["summonerId"])
            
            if result[0]:
                logger.success(f"✅ Success: {result[1]}")
                self.processed_players += 1
            else:
                logger.warning(f"⚠️ Skip: {result[1]}")
                self.skipped_players += 1
                if "summoner not found" in result[1].lower():
                    self.failed_summoner_lookups += 1
    
    def process_player(self, region: str, summoner_id: str) -> Tuple[bool, str]:
        logger.debug("🔍 Looking up summoner...")
        summoner = self.api.get_summoner_by_id(region, summoner_id)
        
        if not summoner or not summoner.get("puuid"):
            logger.debug("❌ Summoner lookup failed")
            return (False, f"Summoner {summoner_id[:6]} not found")
        
        logger.debug(f"📨 Found PUUID: {summoner['puuid'][:8]}...")
        logger.info("🔍 Fetching match history...")
        match_ids = self.api.get_match_history(summoner["puuid"], region)
        
        if not match_ids:
            logger.info("📭 No matches found for summoner")
            return (False, f"No matches for {summoner_id[:6]}")

        valid_matches = []
        logger.info(f"🔍 Analyzing {len(match_ids)} matches...")
        
        for idx, match_id in enumerate(match_ids[:Config.MAX_MATCHES_PER_PLAYER]):
            logger.debug(f"📦 Processing match {idx+1}/{len(match_ids)}")
            match = self.api.get_match_details(match_id, region)
            
            if not match:
                logger.debug("❌ Match details not found")
                continue
                
            if not self.is_valid_match(match, summoner["puuid"]):
                logger.debug("⚠️ Invalid match (player not participating)")
                continue
            
            patch_data = match["info"]["gameVersion"].split('.')[:2]
            patch_data[0] = str(int(patch_data[0]) + 10)
            patch = '.'.join(patch_data)
            
            if patch not in self.active_patches:
                logger.info(f"⏩ Reached outdated patch {patch}, stopping processing")
                break
                
            valid_matches.append(match)
            logger.debug(f"✅ Added valid match from patch {patch}")

        if not valid_matches:
            logger.info("📭 No valid matches remaining after filtering")
            return (False, f"No valid matches for {summoner_id[:6]}")
            
        logger.info(f"📊 Processing {len(valid_matches)} valid matches")
        stats_by_patch = self.process_matches(summoner, valid_matches)
        if stats_by_patch:
            self.aggregate_stats(stats_by_patch)
        return (True, f"Processed {len(valid_matches)} matches")


    def _format_skip_reason(self, skip_details: Dict, player_id: str, puuid: str) -> str:
        reasons = []
        if skip_details["invalid_id"] > 0:
            reasons.append(f"{skip_details['invalid_id']} invalid match IDs")
        if skip_details["match_not_found"] > 0:
            reasons.append(f"{skip_details['match_not_found']} matches not found")
        if skip_details["player_not_in_match"] > 0:
            reasons.append(f"{skip_details['player_not_in_match']} matches without player")
        if skip_details["validation_error"] > 0:
            reasons.append(f"{skip_details['validation_error']} validation errors")
        
        return f"No valid matches - Reasons: {', '.join(reasons)}" if reasons else f"No valid matches for {player_id}"

    def validate_match_id(self, match_id: str, region: str) -> bool:
        try:
            parts = match_id.split('_')
            return (len(parts) == 2 and parts[0] == region.upper() and parts[1].isdigit() and len(parts[1]) == 10)
        except Exception:
            return False

    def process_matches(self, summoner: Dict, matches: List[Dict]) -> Dict[str, Dict[str, ChampionStats]]:
        """Process matches and return stats organized by patch"""
        stats_by_patch = defaultdict(lambda: defaultdict(ChampionStats))
        puuid = summoner["puuid"]
        
        for match in matches:
            try:
                game_version = match["info"]["gameVersion"]
                patch_data = game_version.split('.')[:2]
                patch_data[0] = str(int(patch_data[0]) + 10)
                patch = '.'.join(patch_data)
                
                participants = match["info"]["participants"]
                player = next(p for p in participants if p["puuid"] == puuid)
                
                player_champ = Config.normalize_champion_name(player["championName"])
                opponent = self.get_lane_opponent(player, participants)
                
                if opponent:
                    stats_by_patch[patch][player_champ].add_game(
                        win=player["win"],
                        kills=player["kills"],
                        deaths=player["deaths"],
                        assists=player["assists"],
                        opponent=opponent
                    )
                    
                    opponent_participant = next(
                        p for p in participants 
                        if Config.normalize_champion_name(p["championName"]) == opponent
                    )
                    
                    stats_by_patch[patch][opponent].add_game(
                        win=opponent_participant["win"],
                        kills=opponent_participant["kills"],
                        deaths=opponent_participant["deaths"],
                        assists=opponent_participant["assists"],
                        opponent=player_champ
                    )
                            
            except Exception as e:
                logger.debug(f"Match processing error: {str(e)}")
                continue
                
        return stats_by_patch
    
    def get_lane_opponent(self, player: Dict, participants: List[Dict]) -> Optional[str]:
        try:
            position = player["teamPosition"]
            opponents = [p for p in participants 
                        if p["teamId"] != player["teamId"] 
                        and p.get("teamPosition") == position]
            if not opponents:
                logger.debug(f"No lane opponent for {player['championName']}")
            return Config.normalize_champion_name(opponents[0]["championName"]) if opponents else None
        except Exception:
            return None

    def aggregate_stats(self, stats_by_patch: Dict[str, Dict[str, ChampionStats]]) -> None:
        """Aggregate stats from multiple patches"""
        for patch, patch_stats in stats_by_patch.items():
            for champion, champion_stats in patch_stats.items():
                existing = self.global_stats[patch][champion]
                
                existing.games += champion_stats.games
                existing.wins += champion_stats.wins
                existing.kills += champion_stats.kills
                existing.deaths += champion_stats.deaths
                existing.assists += champion_stats.assists
                
                for opponent, matchup in champion_stats.matchups.items():
                    existing.matchups[opponent]["games"] += matchup["games"]
                    existing.matchups[opponent]["wins"] += matchup["wins"]
                    existing.matchups[opponent]["kills"] += matchup["kills"]
                    existing.matchups[opponent]["deaths"] += matchup["deaths"]
                    existing.matchups[opponent]["assists"] += matchup["assists"]
    
    def save_data(self) -> None:
        """Save data for all patches with atomic writes"""
        logger.info(f"🔄 Saving data for {len(self.active_patches)} patches")
        for patch in self.active_patches:
            self.save_patch_data(patch)

    def save_patch_data(self, patch: str) -> None:
        """Save all data for a specific patch"""
        patch_dir = Config.get_patch_output_dir(patch)
        self.save_global_stats(patch, patch_dir)
        self.save_matchup_stats(patch, patch_dir)


    def save_global_stats(self, patch: str, patch_dir: str) -> None:
        """Merge-and-replace strategy for global stats"""
        file_path = os.path.join(patch_dir, "global_stats.csv")
        
        existing_data = {}
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            existing_data = existing_df.set_index('champion').to_dict('index')

        for champ_name, champ_stats in self.global_stats[patch].items():
            if champ_stats.games <= 0:
                continue
                
            champ = Config.normalize_champion_name(champ_name)
            
            if champ not in existing_data:
                existing_data[champ] = {
                    'games': 0,
                    'wins': 0,
                    'kills': 0,
                    'deaths': 0,
                    'assists': 0
                }

            existing_data[champ]['games'] += champ_stats.games
            existing_data[champ]['wins'] += champ_stats.wins
            existing_data[champ]['kills'] += champ_stats.kills
            existing_data[champ]['deaths'] += champ_stats.deaths
            existing_data[champ]['assists'] += champ_stats.assists

        stats_list = []
        for champ, data in existing_data.items():
            games = data['games']
            if games == 0:
                continue
                
            stats_list.append({
                "champion": champ,
                "games": games,
                "wins": data['wins'],
                "kills": data['kills'],
                "deaths": data['deaths'],
                "assists": data['assists'],
                "win_rate": round((data['wins'] / games) * 100, 2),
                "kda": round((data['kills'] + data['assists']) / max(1, data['deaths']), 2),
                "avg_kills": round(data['kills'] / games, 2),
                "avg_deaths": round(data['deaths'] / games, 2),
                "avg_assists": round(data['assists'] / games, 2),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        if stats_list:
            pd.DataFrame(stats_list).to_csv(file_path, index=False)
            logger.info(f"💾 Updated global stats for patch {patch}")

    def save_matchup_stats(self, patch: str, patch_dir: str) -> None:
        """Merge-and-replace strategy for matchups"""
        matchup_count = 0
        
        for champ_name, champ_stats in self.global_stats[patch].items():
            if champ_stats.games < Config.MIN_GAMES_THRESHOLD:
                continue

            normalized_name = Config.normalize_champion_name(champ_name)
            file_path = os.path.join(patch_dir, "matchups", f"{normalized_name}.csv")
            
            existing_data = {}
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                existing_data = existing_df.set_index('opponent').to_dict('index')

            for opponent, matchup in champ_stats.matchups.items():
                norm_opp = Config.normalize_champion_name(opponent)
                
                if norm_opp not in existing_data:
                    existing_data[norm_opp] = {
                        "games": 0,
                        "wins": 0,
                        "kills": 0,
                        "deaths": 0,
                        "assists": 0
                    }

                existing_data[norm_opp]["games"] += matchup["games"]
                existing_data[norm_opp]["wins"] += matchup["wins"]
                existing_data[norm_opp]["kills"] += matchup["kills"]
                existing_data[norm_opp]["deaths"] += matchup["deaths"]
                existing_data[norm_opp]["assists"] += matchup["assists"]

            matchup_list = []
            for opponent, data in existing_data.items():
                games = data["games"]
                if games == 0:
                    continue
                    
                matchup_list.append({
                    "opponent": opponent,
                    "games": games,
                    "wins": data["wins"],
                    "kills": data["kills"],
                    "deaths": data["deaths"],
                    "assists": data["assists"],
                    "win_rate": round((data["wins"] / games) * 100, 2),
                    "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                    "avg_kills": round(data["kills"] / games, 2),
                    "avg_deaths": round(data["deaths"] / games, 2),
                    "avg_assists": round(data["assists"] / games, 2),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            if matchup_list:
                pd.DataFrame(matchup_list).to_csv(file_path, index=False)
                matchup_count += len(matchup_list)
        
        logger.info(f"💾 Updated {matchup_count} matchups for patch {patch}")

    def log_final_stats(self) -> None:
        total_time = (time.time() - self.start_time) / 60
        logger.info("📊 Final Statistics:")
        
        logger.info(f"📦 Patch Overview:")
        logger.info(f"  - Target patches: {len(self.target_patches)}")
        for patch in sorted(self.target_patches):
            year, num = patch.split('.')
            logger.info(f"    ▪ 20{year} Season Patch {num}")
        
        logger.info(f"  - Existing patches: {len(self.existing_patches)}")
        logger.info(f"  - Newly scraped patches: {len(self.active_patches)}")
        if self.active_patches:
            for patch in sorted(self.active_patches):
                year, num = patch.split('.')
                logger.info(f"    ✨ 20{year} Season Patch {num} (fresh data)")
        
        if self.processed_players == 0:
            logger.warning("🌧️  No players processed - check API key/network")
            return
            
        logger.info(f"\n⏱️  Total runtime: {total_time:.1f} minutes")
        logger.info(f"✅ Players processed: {self.processed_players}")
        logger.info(f"⚠️  Players skipped: {self.skipped_players}")
        logger.info(f"🔍 Failed summoner lookups: {self.failed_summoner_lookups}")
        logger.info(f"📡 API requests: {self.api.total_requests}")
        logger.info(f"❌ Failed requests: {self.api.failed_requests}")
        logger.info(f"🏁 Script completed")

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
    logger.success = lambda msg, *args: logger._log(25, msg, args)
    
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