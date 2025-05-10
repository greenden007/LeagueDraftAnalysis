import os
from typing import List, Tuple, Dict
from heuristic import DraftAnalyst

class DraftAgent:
    def __init__(self, side: str = "blue", patch: str = "latest"):
        """
        Initialize the draft agent
        :param side: 'blue' or 'red' depending on team side
        :param patch: patch version string (e.g., '25.9') or 'latest'
        """
        self.side = side
        self.current_bans: List[str] = []
        self.current_picks: List[Tuple[str, str, str]] = []  
        self.phase_history: List[str] = []
        
        if patch == "latest":
            self.data_dir = self._find_latest_patch()
        else:
            self.data_dir = f"soloq_stats/patch_{patch}"
            
        self.analyst = DraftAnalyst(data_dir=self.data_dir)
        
    def _find_latest_patch(self) -> str:
        """Find the latest patch directory"""
        patches = [d for d in os.listdir("soloq_stats") if d.startswith("patch_")]
        sorted_patches = sorted(patches, key=lambda x: [int(n) for n in x.replace("patch_", "").split(".")], reverse=True)
        return f"soloq_stats/{sorted_patches[0]}"

    def update_draft_state(
        self,
        bans: List[str],
        picks: List[Tuple[str, str, str]],
        phase: str
    ) -> None:
        """
        Update the current draft state
        :param bans: List of banned champions
        :param picks: List of tuples (champion, role, team)
        :param phase: Current draft phase (e.g., 'Ban 1', 'Pick 2')
        """
        self.current_bans = bans
        self.current_picks = picks
        self.phase_history.append(phase)

    def get_suggestion(self) -> Dict:
        """
        Get ban/pick recommendations based on current state
        Returns: {
            'recommended_bans': List[str],
            'recommended_picks': Dict[role: List[str]],
            'counters': Dict[enemy_champ: List[str]],
            'draft_score': float
        }
        """
        analysis = self.analyst.analyze_draft(
            side=self.side,
            bans=self.current_bans,
            picks=self.current_picks,
            phase=self.phase_history[-1] if self.phase_history else "Ban 1"
        )
        
        recs = analysis.get('recommendations', {})
        bans = recs.get('bans', [])
        picks = recs.get('picks', {})
        counters = recs.get('counters', {})
        return {
            'recommended_bans': bans[:3],
            'recommended_picks': picks,
            'counters': counters,
            'draft_score': analysis.get('score', 0)
        }

    def format_suggestion(self, suggestion: Dict) -> str:
        """Convert suggestion dictionary to human-readable format"""
        output = []
        
        score = suggestion['draft_score']
        output.append(f"Current Draft Score: {score:.2f} ({'Favorable' if score > 0 else 'Unfavorable'})")
        
        if suggestion['recommended_bans']:
            output.append("\nRecommended Bans:")
            output.extend([f"- {ban}" for ban in suggestion['recommended_bans']])
        
        if suggestion['recommended_picks']:
            output.append("\nRecommended Picks:")
            for role, champs in suggestion['recommended_picks'].items():
                output.append(f"{role}:")
                output.extend([f"  - {champ}" for champ in champs[:3]])
                
        if suggestion['counters']:
            output.append("\nEnemy Counters:")
            for enemy, counters in suggestion['counters'].items():
                output.append(f"Against {enemy}:")
                output.extend([f"  - {c}" for c in counters[:3]])
                
        return "\n".join(output)

if __name__ == "__main__":
    agent = DraftAgent(side="blue")
    
    agent.update_draft_state(
        bans=["Zac", "Kaisa"],
        picks=[
            ("Jinx", "ADC", "blue"),
            ("Lulu", "Support", "blue"),
            ("Draven", "ADC", "red")
        ],
        phase="Ban 2"
    )
    
    suggestion = agent.get_suggestion()
    print(agent.format_suggestion(suggestion))