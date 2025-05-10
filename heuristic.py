"""
Advanced Draft Phase Analysis System
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)

class Config:
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
        """Convert any champion name format to filename-safe version"""
        # Normalize input to mapping key
        clean_name = name.strip().lower().replace("'", "").replace(" ", "").replace(".", "")
        if clean_name in cls.CHAMPION_MAPPING:
            return cls.CHAMPION_MAPPING[clean_name]
        # Fallback for unknown names
        return name.title().replace("'", "").replace(" ", "").replace(".", "")

class DraftAnalyst:
    def __init__(self, data_dir: str = "soloq_stats"):
        self.data_dir = data_dir
        self.global_stats = None
        self.matchup_data = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self._init_role_data()
        self._init_synergy_data()
        self._load_matchup_data()
        self.model = None
        self.phase_weights = self._init_phase_weights()
        self._train_model()

    def _init_role_data(self):
        """Load role viability from JSON file"""
        try:
            with open("role_viability.json", 'r') as f:
                self.role_viability = json.load(f)
            logger.info("Loaded role viability data")
        except Exception as e:
            logger.error("Failed to load role data: %s", str(e))
            raise

    def _init_synergy_data(self):
        """Role-based synergy matrix"""
        self.synergy = {
            # ADC-Support Pairs
            'ADC-Support': {
                ('Jinx', 'Lulu'): 0.97,
                ('KogMaw', 'Lulu'): 0.99,
                ('Vayne', 'Janna'): 0.93,
                ('Draven', 'Thresh'): 0.98,
                ('Samira', 'Leona'): 0.99,
                ('Ezreal', 'Yuumi'): 0.96,
                ('Aphelios', 'Thresh'): 0.95,
                ('Zeri', 'Lulu'): 0.94,
                ('Kalista', 'Alistar'): 0.97,
                ('Varus', 'Braum'): 0.92,
                ('Twitch', 'Yuumi'): 0.95,
                ('Kaisa', 'Nautilus'): 0.93,
                ('Xayah', 'Rakan'): 0.99,
                ('Jhin', 'Morgana'): 0.91,
                ('Caitlyn', 'Lux'): 0.94
            },
            
            # Jungle-Mid Synergies
            'Jungle-Mid': {
                ('Elise', 'Syndra'): 0.95,
                ('Lee Sin', 'Orianna'): 0.97,
                ('Zac', 'Yasuo'): 0.98,
                ('Ekko', 'Zed'): 0.93,
                ('Hecarim', 'Azir'): 0.94,
                ('Kayn', 'Akali'): 0.92,
                ('Diana', 'Yone'): 0.96,
                ('Nocturne', 'Twisted Fate'): 0.99,
                ('Pantheon', 'Galio'): 0.95,
                ('Vi', 'Qiyana'): 0.93
            },
            
            # Jungle-Top Synergies
            'Jungle-Top': {
                ('Camille', 'Jarvan IV'): 0.96,
                ('Malphite', 'Zac'): 0.97,
                ('Shen', 'Nocturne'): 0.98,
                ('Ornn', 'Sejuani'): 0.95,
                ('Fiora', 'Lee Sin'): 0.93,
                ('Darius', 'Hecarim'): 0.94,
                ('Gnar', 'Xin Zhao'): 0.92,
                ('Aatrox', 'Kayn'): 0.91
            },
            
            # Mid-Support Synergies
            'Mid-Support': {
                ('Yasuo', 'Alistar'): 0.95,
                ('Zoe', 'Thresh'): 0.92,
                ('Lux', 'Morgana'): 0.96,
                ('Katarina', 'Lulu'): 0.94,
                ('Azir', 'Braum'): 0.93,
                ('Veigar', 'Blitzcrank'): 0.91,
                ('Syndra', 'Leona'): 0.90
            },
            
            # Global Ult Synergies
            'Global': {
                ('Nocturne', 'Shen'): 0.99,
                ('Twisted Fate', 'Pantheon'): 0.98,
                ('Galio', 'Camille'): 0.97,
                ('Zilean', 'Kennen'): 0.96
            },
            
            # Counter-Engage Synergies
            'Counter-Engage': {
                ('Janna', 'Azir'): 0.95,
                ('Gragas', 'Orianna'): 0.94,
                ('Thresh', 'Lee Sin'): 0.93
            }
        }

    def _load_matchup_data(self):
        try:
            self.global_stats = pd.read_csv(f"{self.data_dir}/global_stats.csv")
            # Scale win_rate and kda, then combine into single meta_score
            scaled = self.scaler.fit_transform(
                self.global_stats[['win_rate', 'kda']].values
            )
            self.global_stats['meta_score'] = np.mean(scaled, axis=1)
            
            for champ in self.global_stats['champion'].unique():
                try:
                    # Use normalized name for file lookup
                    normalized = Config.normalize_champion_name(champ)
                    df = pd.read_csv(f"{self.data_dir}/matchups/{normalized}.csv")
                    self.matchup_data[champ] = df.set_index('opponent')['win_rate'].to_dict()
                except FileNotFoundError:
                    continue
            
            logger.info("Loaded matchup data for %d champions", len(self.matchup_data))
        except Exception as e:
            logger.error("Failed to load data: %s", str(e))
            raise

    def _init_phase_weights(self):
        return {
            'Ban 1': 0.1, 'Pick 1': 0.15, 'Ban 2': 0.2, 'Pick 2': 0.25,
            'Ban 3': 0.3, 'Pick 3': 0.35, 'Final Ban': 0.4, 'Final Pick': 0.45
        }

    def _train_model(self):
        self.model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
        # Match feature dimension to number of components (phase_weights length)
        dim = len(self.phase_weights)
        X = np.random.rand(1000, dim)
        y = np.random.rand(1000) * 2 - 1
        self.model.fit(X, y)

    def analyze_draft(
        self,
        side: str,
        bans: List[str],
        picks: List[Tuple[str, str, str]],  # (champion, role, team)
        phase: str
    ) -> Dict[str, float]:
        """
        Full draft analysis with future prediction
        Returns: {
            'score': -1 to 1,
            'components': { ... },
            'recommendations': { ... }
        }
        """
        try:
            # Preprocess inputs
            processed_picks = [(Config.normalize_champion_name(p[0]), p[1], p[2]) for p in picks]
            ally_picks = [p for p in processed_picks if p[2] == side]
            enemy_picks = [p for p in processed_picks if p[2] != side]
            
            # Calculate components
            components = {
                'current_strength': self._current_strength(ally_picks, enemy_picks),
                'predicted_strength': self._predict_future_strength(ally_picks, enemy_picks, phase),
                'synergy_score': self._calculate_synergy(ally_picks),
                'counter_score': self._calculate_counter_score(ally_picks, enemy_picks),
                'ban_efficiency': self._ban_efficiency(bans, enemy_picks),
                'role_viability': self._role_viability_score(ally_picks + enemy_picks),
                'flex_potential': self._flex_potential(ally_picks),
                'phase_risk': self.phase_weights.get(phase, 0.5)
            }
            
            # ML prediction
            ml_score = self.model.predict([list(components.values())])[0]
            
            # Combine scores
            weights = np.array([0.25, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05])
            scores = np.array(list(components.values()))
            final_score = np.clip(np.dot(weights, scores) * 0.7 + ml_score * 0.3, -1, 1)
            
            return {
                'score': final_score,
                'components': components,
                'recommendations': self._generate_recommendations(ally_picks, enemy_picks, phase)
            }
            
        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            return {'score': 0, 'components': {}, 'recommendations': {}}

    def _current_strength(self, ally_picks: list, enemy_picks: list) -> float:
        ally_meta = self._get_meta_score([p[0] for p in ally_picks])
        enemy_meta = self._get_meta_score([p[0] for p in enemy_picks])
        return ally_meta - enemy_meta

    def _predict_future_strength(self, ally_picks: list, enemy_picks: list, phase: str) -> float:
        # Determine remaining phases index, default to 0 if unknown phase
        phases = list(self.phase_weights.keys())
        remaining_phases = phases.index(phase) if phase in phases else 0
        predicted_ally = self._predict_optimal_picks(ally_picks, enemy_picks, remaining_phases)
        predicted_enemy = self._predict_optimal_picks(enemy_picks, ally_picks, remaining_phases)
        return self._get_meta_score(predicted_ally) - self._get_meta_score(predicted_enemy)

    def _calculate_synergy(self, picks: list) -> float:
        score = 0
        champions = [p[0] for p in picks]
        roles = [p[1] for p in picks]
        
        # Check role-based synergies
        for synergy_type in self.synergy.values():
            for (c1, c2), value in synergy_type.items():
                if c1 in champions and c2 in champions:
                    score += value
                    
        # Check lane synergies
        adc = next((p[0] for p in picks if p[1] == 'ADC'), None)
        support = next((p[0] for p in picks if p[1] == 'Support'), None)
        if adc and support:
            score += self.synergy['ADC-Support'].get((adc, support), 0)
            
        return np.tanh(score / 10)  # Normalize to 0-1

    def _calculate_counter_score(self, ally_picks: list, enemy_picks: list) -> float:
        score = 0
        for a_champ, a_role, _ in ally_picks:
            for e_champ, e_role, _ in enemy_picks:
                if a_role == e_role:
                    win_rate = self.matchup_data.get(a_champ, {}).get(e_champ, 50)
                    score += (win_rate - 50) / 50
        return score / max(len(ally_picks)*len(enemy_picks), 1)

    def _ban_efficiency(self, bans: list, enemy_picks: list) -> float:
        enemy_roles = [p[1] for p in enemy_picks]
        candidates = []
        
        for role in set(enemy_roles):
            role_champs = self.role_viability.get(role, [])
            for c in role_champs:
                counter_score = sum(
                    self.matchup_data.get(c, {}).get(a[0], 50) - 50
                    for a in enemy_picks
                )
                if counter_score > 20:
                    candidates.append((c, counter_score))
                    
        return (len(candidates) * 0.1) + (sum(c[1] for c in candidates) * 0.2)

    def _role_viability_score(self, picks: list) -> float:
        valid = sum(1 for p in picks if p[0] in self.role_viability.get(p[1], []))
        return valid / len(picks) if picks else 0

    def _flex_potential(self, picks: list) -> float:
        flex_counts = [sum(1 for r in self.role_viability.values() if c in r) 
                      for c, _, _ in picks]
        return np.mean(flex_counts) / 5

    def _get_meta_score(self, champions: list) -> float:
        scores = []
        for c in champions:
            if c in self.global_stats['champion'].values:
                scores.append(self.global_stats[self.global_stats['champion'] == c]['meta_score'].values[0])
        return np.mean(scores) if scores else 0

    def _predict_optimal_picks(self, current_picks: list, enemy_picks: list, remaining_phases: int) -> list:
        needed_roles = self._missing_roles(current_picks)
        candidates = []
        
        for role in needed_roles:
            role_candidates = self.role_viability.get(role, [])
            best_pick = max(
                role_candidates,
                key=lambda c: self._pick_value(c, role, current_picks, enemy_picks),
                default=None
            )
            if best_pick:
                candidates.append(best_pick)
                
        return current_picks + candidates[:remaining_phases]

    def _pick_value(self, champion: str, role: str, ally_picks: list, enemy_picks: list) -> float:
        meta_score = self.global_stats[self.global_stats['champion'] == champion]['meta_score'].values[0]
        counter_score = sum(
            self.matchup_data.get(champion, {}).get(e[0], 50) - 50 
            for e in enemy_picks
            if e[1] == role
        )
        synergy_score = sum(
            self.synergy['ADC-Support'].get((champion, a[0]), 0) 
            if role == 'ADC' and a[1] == 'Support' else 0
            for a in ally_picks
        )
        return meta_score + (counter_score / 100) + synergy_score

    def _generate_recommendations(self, ally_picks: list, enemy_picks: list, phase: str) -> dict:
        return {
            'bans': self._recommend_bans(ally_picks, enemy_picks),
            'picks': self._recommend_picks(ally_picks, enemy_picks),
            'counters': self._recommend_counters(enemy_picks)
        }

    def _recommend_bans(self, ally_picks: list, enemy_picks: list) -> list:
        enemy_roles = [p[1] for p in enemy_picks]
        candidates = []
        
        for role in set(enemy_roles):
            role_champs = self.role_viability.get(role, [])
            for c in role_champs:
                counter_score = sum(
                    self.matchup_data.get(c, {}).get(a[0], 50) - 50
                    for a in ally_picks
                )
                if counter_score > 20:
                    candidates.append((c, counter_score))
                    
        return [c[0] for c in sorted(candidates, key=lambda x: -x[1])[:3]]

    def _recommend_picks(self, ally_picks: list, enemy_picks: list) -> dict:
        needed_roles = self._missing_roles(ally_picks)
        recommendations = {}
        
        for role in needed_roles:
            candidates = self.role_viability.get(role, [])
            scored = [
                (c, self._pick_value(c, role, ally_picks, enemy_picks))
                for c in candidates
            ]
            recommendations[role] = [c[0] for c in sorted(scored, key=lambda x: -x[1])[:3]]
            
        return recommendations

    def _recommend_counters(self, enemy_picks: list) -> dict:
        counters = {}
        for champ, role, _ in enemy_picks:
            role_counters = self.role_viability.get(role, [])
            counter_scores = [
                (c, self.matchup_data.get(c, {}).get(champ, 50))
                for c in role_counters
            ]
            counters[champ] = [c[0] for c in sorted(counter_scores, key=lambda x: -x[1])[:3]]
        return counters

    def _missing_roles(self, picks: list) -> list:
        return [r for r in self.role_viability if not any(p[1] == r for p in picks)]

    def _is_counter_to_team(self, champion: str, picks: list) -> bool:
        return any(
            self.matchup_data.get(champion, {}).get(p[0], 50) > 55
            for p in picks
        )