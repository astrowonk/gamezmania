import json
import pandas as pd


class Gamezmania():
    """read and parse a json from cardzmania"""
    def __init__(self, filename) -> None:
        with open(filename, 'r') as j:
            self.raw_data = json.load(j)

        self.name_map = self.make_name_map()
        self.score_df = self.make_score_dataframe()

    def make_name_map(self):
        """Map the order to the player id """
        return {i: x['id'] for i, x in enumerate(self.raw_data['g']['i']['p'])}

    def make_score_dataframe(self):
        """make a datafraem of the scores"""
        out = []

        for r, record in enumerate(self.raw_data['g']['r']):
            out.extend([{
                'round': r,
                'player': self.name_map[i],
                'points': x.get('pts', 0)
            } for i, x in enumerate(record['p'])])

        out.extend([{
            'round': r + 1,
            'player': self.name_map[i],
            'points': x.get('pts', 0)
        } for i, x in enumerate(self.raw_data['g']['result']['p'])])
        return pd.DataFrame(out)