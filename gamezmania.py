import json
import pandas as pd
import hashlib
from collections import deque

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

    def __init__(self, filename, custom_player_map=None) -> None:
        with open(filename, 'r') as j:
            self.raw_data = json.load(j)
        self.file_name = filename
        self.custom_player_map = custom_player_map

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
        hand_number = None
        for row_num, row in enumerate(better_data):

            if 'c' in row:
                the_round += 1
                first_player = 0 + the_round - 1
                hand_number = 1
                for i, bid in enumerate(row['c']):
                    new_dict = {}
                    new_dict['round'] = the_round
                    new_dict['player'] = self.name_map[i]
                    new_dict['bid'] = bid
                    out.append(new_dict)

            elif 'pl' in row and row['pl']:
                card_data = deque(row['pl'])
                card_order = deque(range(1, n_players + 1))
                # print(f"Rotating {first_player} + {the_round - 1}")
                card_data.rotate(first_player)
                card_order.rotate(first_player)
                card_order = list(card_order)

                for i, card in enumerate(card_data):
                    new_dict = {}
                    new_dict['player'] = self.name_map[i]
                    rank, suit = self.parse_card_string(card)
                    new_dict['card_rank'] = rank
                    new_dict['card_suit'] = suit
                    new_dict['card_order'] = card_order[i]
                    new_dict['round'] = the_round
                    new_dict['hand'] = hand_number
                    out.append(new_dict)
                first_player = row['g']
                hand_number += 1

            elif 'tram' in row:
                for player, cards in enumerate(row['tram']):
                    for i, card in enumerate(cards):
                        new_dict = {}
                        new_dict['player'] = self.name_map[player]
                        rank, suit = self.parse_card_string(card)
                        new_dict['card_rank'] = rank
                        new_dict['card_suit'] = suit
                        new_dict['round'] = the_round
                        new_dict['tram'] = True
                        out.append(new_dict)
                is_new_round = False

        df = pd.DataFrame(out).convert_dtypes('pyarrow')
        if 'tram' in df.columns:
            df['tram'] = df['tram'].fillna(False)
        else:
            df['tram'] = False

        df['bad_card'] = df['card_rank'].isin(['9', '8', 'T', '7',
                                               'J']).astype(bool)
        self.make_trump_map()
        df['is_trump'] = df.assign(
            map_key=lambda x: x['round'].astype('str') + '_' + x['card_suit']
        )['map_key'].apply(lambda x: self.trump_map.get(x, False))
        df.loc[df['is_trump'] == True, 'bad_card'] = False

        df['unique_hash'] = hashlib.sha224(
            self.raw_data['g']['c'].encode()).hexdigest()[:20]
        if self.custom_player_map:
            df['player'] = df['player'].apply(
                lambda x: self.custom_player_map.get(x, x))
        df['file_name'] = self.file_name
        df['ace_flag'] = (df['card_rank'] == 'ace')
        return df

    def make_trump_map(self):
        self.trump_map = {}
        for round, row in enumerate(self.raw_data['g']['r']):
            rank, suit = self.parse_card_string(row['tp'])
            self.trump_map[f"{round + 1}_{suit}"] = True

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

        df = pd.DataFrame(out)
        if self.custom_player_map:
            df['player'] = df['player'].apply(
                lambda x: self.custom_player_map.get(x, x))
        return df
