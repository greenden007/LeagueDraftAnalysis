"""
Riot API Scraper - Global Aggregated Stats with Matchups
"""

import os
import time
import logging
import socket
import sys
from datetime import datetime
from collections import defaultdict, deque
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple

# ========================
# Configuration
# ========================
class Config:
    """Centralized configuration with validation"""
    API_KEY = os.getenv("RIOT_API_KEY")
    BASE_URL = "https://{region}.api.riotgames.com/lol"
    
    REGIONS = ["na1", "euw1", "kr", "eun1", "br1", "la1"]
    MATCH_REGION_MAP = {
        region: "americas" if region in ["na1", "br1", "la1", "la2", "oc1"] 
        else "europe" if region in ["euw1", "eun1", "tr1", "ru"] 
        else "asia" 
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

    @classmethod
    def normalize_champion_name(cls, name: str) -> str:
        return cls.CHAMPION_MAPPING.get(name.lower(), name.title())

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
    OUTPUT_DIR = "soloq_stats"
    MATCHUP_DIR = os.path.join(OUTPUT_DIR, "matchups")
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 3]

    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("RIOT_API_KEY environment variable not set")
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MATCHUP_DIR, exist_ok=True)

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

    def add_game(self, win: bool, kills: int, deaths: int, assists: int, opponent_champ: Optional[str] = None) -> None:
        self.games += 1
        self.wins += int(win)
        self.kills += kills
        self.deaths += deaths
        self.assists += assists
        
        if opponent_champ:
            opponent = Config.normalize_champion_name(opponent_champ)
            self.matchups[opponent]["games"] += 1
            self.matchups[opponent]["wins"] += int(win)
            self.matchups[opponent]["kills"] += kills
            self.matchups[opponent]["deaths"] += deaths
            self.matchups[opponent]["assists"] += assists

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
        self.last_connection_state = True  # Track connection state changes

    def check_connectivity(self) -> None:
        """Monitor internet connection with state-aware logging"""
        while True:
            try:
                # Test connection to Cloudflare DNS
                socket.create_connection(("1.1.1.1", 53), timeout=5)
                
                # Only log restoration if state changed
                if not self.last_connection_state:
                    logger.info("🌐✅ Internet connection restored! Resuming data collection...")
                    self.last_connection_state = True
                
                self.connectivity_checked = True
                return
            
            except OSError:
                # Log initial disconnection
                if self.last_connection_state:
                    logger.warning("🌐❌ Internet connection lost! Pausing until restored...")
                    self.last_connection_state = False
                else:
                    # Periodic waiting message
                    logger.info("🌐⏳ Still waiting for internet connection...")
                
                time.sleep(10)

    def get_json(self, url: str, endpoint_type: str) -> Optional[Dict]:
        """Make API request with connection-aware error handling"""
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
                    logger.warning(f"⏳ Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                self.failed_requests += 1
                logger.debug(f"🌐🔌 Connection error: {str(e)}")
                self.check_connectivity()  # Re-check connectivity
                continue
            except requests.exceptions.RequestException as e:
                self.failed_requests += 1
                logger.debug(f"⚠️ Attempt {attempt+1} failed: {str(e)}")
                time.sleep(Config.RETRY_DELAYS[attempt])
                
        logger.warning(f"⚠️ Failed after {Config.MAX_RETRIES} attempts for {url}")
        return None

    # Keep existing API methods unchanged below
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
        self.global_stats = defaultdict(ChampionStats)
        self.start_time = time.time()
        self.processed_players = 0
        self.skipped_players = 0
        self.failed_summoner_lookups = 0

    def run(self) -> None:
        logger.info("🌍 Starting global data collection")
        try:
            for region in Config.REGIONS:
                self.process_region(region)
                self.save_data()
            
            self.log_final_stats()
        except KeyboardInterrupt:
            logger.info("🛑 Interrupt received. Finalizing data...")
            self.save_data()
            self.log_final_stats()
            sys.exit(0)
        except Exception as e:
            logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
            raise
        finally:
            self.save_data()

    def process_region(self, region: str) -> None:
        logger.info(f"🏆 Processing {region.upper()}")
        ladder = self.api.get_challenger_league(region)
        if not ladder or not isinstance(ladder, dict) or "entries" not in ladder:
            logger.error(f"Invalid ladder data for {region}")
            return
            
        players = ladder["entries"]
        logger.info(f"Found {len(players)} players in {region.upper()}")
        
        for i, player in enumerate(players):
            logger.info(f"⏳ Processing player {i+1}/{len(players)} in {region.upper()}")
            result = self.process_player(region, player["summonerId"])
            status = "✅ Processed" if result[0] else f"❌ Skipped: {result[1]}"
            logger.info(f"Status: {status}")
            
            if result[0]:
                self.processed_players += 1
            else:
                self.skipped_players += 1
                if "summoner not found" in result[1].lower():
                    self.failed_summoner_lookups += 1
            
            time.sleep(0.1)

    def process_player(self, region: str, summoner_id: str) -> Tuple[bool, str]:
        summoner = self.api.get_summoner_by_id(region, summoner_id)
        if not summoner:
            return (False, f"Summoner not found by ID {summoner_id[:6]}")
        
        puuid = summoner.get("puuid")
        if not puuid:
            return (False, f"Summoner {summoner_id[:6]} has no PUUID")
        
        match_ids = self.api.get_match_history(puuid, region)
        if not match_ids:
            return (False, f"No match history for {summoner_id[:6]}")

        valid_matches = []
        skip_details = {"invalid_id": 0, "match_not_found": 0, "player_not_in_match": 0, "validation_error": 0}
        
        for match_id in match_ids[:Config.MAX_MATCHES_PER_PLAYER]:
            if not self.validate_match_id(match_id, region):
                skip_details["invalid_id"] += 1
                continue
                
            match = self.api.get_match_details(match_id, region)
            if not match:
                skip_details["match_not_found"] += 1
                continue
                
            try:
                if not any(p["puuid"] == puuid for p in match["info"]["participants"]):
                    skip_details["player_not_in_match"] += 1
                    continue
            except Exception:
                skip_details["validation_error"] += 1
                continue
                
            valid_matches.append(match)
        
        if not valid_matches:
            return (False, self._format_skip_reason(skip_details, summoner_id[:6], puuid[:8]))
        
        try:
            stats = self.process_matches(summoner, valid_matches)
            self.aggregate_stats(stats)
            return (True, f"Processed {len(valid_matches)} matches")
        except Exception as e:
            return (False, f"Processing error: {str(e)}")

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

    def process_matches(self, summoner: Dict, matches: List[Dict]) -> Dict[str, ChampionStats]:
        stats = defaultdict(ChampionStats)
        puuid = summoner["puuid"]
        
        for match in matches:
            try:
                participants = match["info"]["participants"]
                player = next(p for p in participants if p["puuid"] == puuid)
                
                # Process both player and their opponent's perspective
                player_champ = Config.normalize_champion_name(player["championName"])
                opponent = self.get_lane_opponent(player, participants)
                
                if opponent:
                    # Add matchup from player's perspective
                    stats[player_champ].add_game(
                        win=player["win"],
                        kills=player["kills"],
                        deaths=player["deaths"],
                        assists=player["assists"],
                        opponent_champ=opponent
                    )
                    
                    # Find opponent participant and add reciprocal matchup
                    opponent_participant = next(
                        p for p in participants 
                        if Config.normalize_champion_name(p["championName"]) == opponent
                    )
                    
                    # Add matchup from opponent's perspective
                    stats[opponent].add_game(
                        win=opponent_participant["win"],
                        kills=opponent_participant["kills"],
                        deaths=opponent_participant["deaths"],
                        assists=opponent_participant["assists"],
                        opponent_champ=player_champ
                    )
                    
            except Exception as e:
                logger.debug(f"Match processing error: {str(e)}")
                continue
                
        return stats

    def get_lane_opponent(self, player: Dict, participants: List[Dict]) -> Optional[str]:
        try:
            position = player["teamPosition"]
            if position in ["TOP", "MID", "JUNGLE", "BOTTOM", "UTILITY"]:
                opponents = [p for p in participants 
                            if p["teamId"] != player["teamId"] 
                            and p.get("teamPosition") == position]
                return Config.normalize_champion_name(opponents[0]["championName"]) if opponents else None
        except Exception:
            return None
        return None

    def aggregate_stats(self, stats: Dict[str, ChampionStats]) -> None:
        for champion, champion_stats in stats.items():
            existing = self.global_stats[champion]
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
        self.save_global_stats()
        self.save_matchup_stats()
        self.global_stats.clear()

    def save_global_stats(self) -> None:
        file_path = os.path.join(Config.OUTPUT_DIR, "global_stats.csv")
        stats_list = []

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            stats_list = existing_df.to_dict('records')

        for champ_name, champ_stats in self.global_stats.items():
            normalized_name = Config.normalize_champion_name(champ_name)
            entry = next((x for x in stats_list if x["champion"] == normalized_name), None)
            
            if not entry:
                stats_list.append({
                    "champion": normalized_name,
                    "games": champ_stats.games,
                    "wins": champ_stats.wins,
                    "win_rate": round(champ_stats.win_rate, 2),
                    "avg_kills": round(champ_stats.kills / champ_stats.games, 2),
                    "avg_deaths": round(champ_stats.deaths / champ_stats.games, 2),
                    "avg_assists": round(champ_stats.assists / champ_stats.games, 2),
                    "kda": round(champ_stats.kda, 2),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                total_games = entry["games"] + champ_stats.games
                entry["games"] = total_games
                entry["wins"] += champ_stats.wins
                entry["avg_kills"] = round(
                    (entry["avg_kills"] * entry["games"] + champ_stats.kills) / total_games, 2)
                entry["avg_deaths"] = round(
                    (entry["avg_deaths"] * entry["games"] + champ_stats.deaths) / total_games, 2)
                entry["avg_assists"] = round(
                    (entry["avg_assists"] * entry["games"] + champ_stats.assists) / total_games, 2)
                entry["win_rate"] = round((entry["wins"] / total_games) * 100, 2)
                entry["kda"] = round((entry["avg_kills"] + entry["avg_assists"]) / max(1, entry["avg_deaths"]), 2)
                entry["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pd.DataFrame(stats_list).to_csv(file_path, index=False)
        logger.info(f"💾 Updated global stats with {len(stats_list)} champions")

    def save_matchup_stats(self) -> None:
        for champ_name, champ_stats in self.global_stats.items():
            normalized_name = Config.normalize_champion_name(champ_name)
            file_path = os.path.join(Config.MATCHUP_DIR, f"{normalized_name}.csv")
            
            existing_data = {}
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                existing_data = existing_df.set_index('opponent').to_dict('index')

            for opponent, data in champ_stats.matchups.items():
                norm_opponent = Config.normalize_champion_name(opponent)
                if norm_opponent in existing_data:
                    existing_data[norm_opponent]["games"] += data["games"]
                    existing_data[norm_opponent]["wins"] += data["wins"]
                    existing_data[norm_opponent]["kills"] += data["kills"]
                    existing_data[norm_opponent]["deaths"] += data["deaths"]
                    existing_data[norm_opponent]["assists"] += data["assists"]
                else:
                    existing_data[norm_opponent] = {
                        "games": data["games"],
                        "wins": data["wins"],
                        "kills": data["kills"],
                        "deaths": data["deaths"],
                        "assists": data["assists"],
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

            matchup_list = []
            for opponent, data in existing_data.items():
                matchup_list.append({
                    "opponent": opponent,
                    "games": data["games"],
                    "win_rate": round((data["wins"] / data["games"]) * 100, 2),
                    "kda": round((data["kills"] + data["assists"]) / max(1, data["deaths"]), 2),
                    "avg_kills": round(data["kills"] / data["games"], 2),
                    "avg_deaths": round(data["deaths"] / data["games"], 2),
                    "avg_assists": round(data["assists"] / data["games"], 2),
                    "last_updated": data.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                })

            if matchup_list:
                pd.DataFrame(matchup_list).to_csv(file_path, index=False)
                logger.debug(f"💾 Updated matchups for {normalized_name} ({len(matchup_list)} matchups)")

    def log_final_stats(self) -> None:
        total_time = (time.time() - self.start_time) / 60
        logger.info("\n📊 Final Statistics:")
        logger.info(f"⏱️  Total runtime: {total_time:.1f} minutes")
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
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler("scraper.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    logger = setup_logging()
    Config.validate()
    
    try:
        scraper = LeagueScraper()
        scraper.run()
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
        raise