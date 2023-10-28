import json
import pandas as pd

MAPPING_DICT = {
    'C': 'clubs',
    'D': 'diamonds',
    'H': 'hearts',
    'S': 'spades',
    'J': 'jack',
    'A': 'ace',
    'Q': 'queen',
    'K': 'king',
}


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

    @staticmethod
    def parse_card_string(card):
        rank = MAPPING_DICT.get(card[0], card[0])
        suit = MAPPING_DICT.get(card[1])
        return rank, suit

    def parse_oh_data(self):
        n_players = len(self.name_map)
        out = []
        better_data = self.raw_data['g']['n']
        is_new_round = False
        the_round = 0
        for row in better_data:

            if 'c' in row:
                if not is_new_round:
                    the_round += 1
                    is_new_round = True
                for i, bid in enumerate(row['c']):
                    new_dict = {}
                    new_dict['round'] = the_round
                    new_dict['player'] = self.name_map[i]
                    new_dict['bid'] = bid
                    out.append(new_dict)

            elif 'pl' in row and row['pl']:
                for i, card in enumerate(row['pl']):
                    new_dict = {}
                    new_dict['player'] = self.name_map[i]
                    rank, suit = self.parse_card_string(card)
                    new_dict['card_rank'] = rank
                    new_dict['card_suit'] = suit
                    new_dict['round'] = the_round
                    is_new_round = False
                    out.append(new_dict)
            elif 'tram' in row:
                for player, cards in enumerate(row['tram']):
                    for i, card in enumerate(cards):
                        new_dict = {}
                        new_dict['player'] = self.name_map[player]
                        rank, suit = self.parse_card_string(card)
                        new_dict['card_rank'] = rank
                        new_dict['card_suit'] = suit
                        new_dict['round'] = the_round + 1 + i
                        new_dict['tram'] = True
                        out.append(new_dict)

        df = pd.DataFrame(out)
        df['tram'] = df['tram'].fillna(False)

        df['bad_card'] = df['card_rank'].isin(['9', '8', 'T',
                                               'J']).astype(bool)
        #df['unique_game_id'] = self.raw_data['v']
        return df

    def make_trump_map(self):
        self.trump_map = {}
        for round, row in enumerate(self.raw_data['g']['r']):
            rank, suit = self.parse_card_string(row['tp'])
            self.trump_map[f"{round + 1}_{suit}"] = True

    def parse_wrong_oh_data(self):
        n_players = len(self.name_map)
        out = []
        round_data = self.raw_data['g']['c']
        is_new_round = False
        the_round = 0
        for i, res in enumerate(round_data.split(',')):
            if res == '-1':
                break
            new_dict = {}
            new_dict['player'] = self.name_map[(i % n_players)]
            if res.isdigit():
                new_dict['bid'] = int(res)
                if not is_new_round:
                    the_round += 1
                    is_new_round = True
                new_dict['round'] = the_round

            else:
                rank, suit = self.parse_card_string(res)
                new_dict['card_rank'] = rank
                new_dict['card_suit'] = suit
                new_dict['round'] = the_round
                is_new_round = False
            out.append(new_dict)
        df = pd.DataFrame(out)
        df['bad_card'] = df['card_rank'].isin(['9', '8', 'T',
                                               'J']).astype(bool)

        return df

    def make_score_dataframe(self, player_map=None):
        """make a datafraem of the scores"""

        if not player_map:
            player_map = {}
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

        df = pd.DataFrame(out)
        if player_map:
            df['player'] = df['player'].map(player_map)
        return df
