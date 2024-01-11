import pandas as pd
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sqlalchemy.exc import OperationalError
from tqdm.notebook import tqdm
import statsmodels.api as sm

COLS_TRAIN = [
    'is_trump',
    'card_rank_hearts_2',
    'card_rank_hearts_3',
    'card_rank_hearts_4',
    'card_rank_hearts_5',
    'card_rank_hearts_6',
    'card_rank_hearts_7',
    'card_rank_hearts_8',
    'card_rank_hearts_9',
    'card_rank_hearts_T',
    'card_rank_hearts_ace',
    'card_rank_hearts_jack',
    'card_rank_hearts_king',
    'card_rank_hearts_queen',
    'card_rank_clubs_2',
    'card_rank_clubs_3',
    'card_rank_clubs_4',
    'card_rank_clubs_5',
    'card_rank_clubs_6',
    'card_rank_clubs_7',
    'card_rank_clubs_8',
    'card_rank_clubs_9',
    'card_rank_clubs_T',
    'card_rank_clubs_ace',
    'card_rank_clubs_jack',
    'card_rank_clubs_king',
    'card_rank_clubs_queen',
    'card_rank_diamonds_2',
    'card_rank_diamonds_3',
    'card_rank_diamonds_4',
    'card_rank_diamonds_5',
    'card_rank_diamonds_6',
    'card_rank_diamonds_7',
    'card_rank_diamonds_8',
    'card_rank_diamonds_9',
    'card_rank_diamonds_T',
    'card_rank_diamonds_ace',
    'card_rank_diamonds_jack',
    'card_rank_diamonds_king',
    'card_rank_diamonds_queen',
    'card_rank_spades_2',
    'card_rank_spades_3',
    'card_rank_spades_4',
    'card_rank_spades_5',
    'card_rank_spades_6',
    'card_rank_spades_7',
    'card_rank_spades_8',
    'card_rank_spades_9',
    'card_rank_spades_T',
    'card_rank_spades_ace',
    'card_rank_spades_jack',
    'card_rank_spades_king',
    'card_rank_spades_queen',
    'bid_div_total_cards',
    'total_cards',
    'total_bid_minus_total_cards',
    'card_order',
    'trump_2',
    'trump_3',
    'trump_4',
    'trump_5',
    'trump_6',
    'trump_7',
    'trump_8',
    'trump_9',
    'trump_T',
    'trump_ace',
    'trump_jack',
    'trump_king',
    'trump_queen',
    'singles_2',
    'singles_3',
    'singles_4',
    'singles_5',
    'singles_6',
    'singles_7',
    'singles_8',
    'singles_9',
    'singles_T',
    'singles_ace',
    'singles_jack',
    'singles_king',
    'singles_queen',
    'n_decks',
    'player_count',
]


def make_delta_columns(df, column1, column2):
    """Add a column that is the difference between two columns.

    Usage:

    make_delta_column(df,'sale_price','forecast_value').

    The new column in this example will be 'sale_price_minus_forecast_value'.

    The function returns this new column name.
    """
    df.loc[:, column1 + '_minus_' +
           column2] = (df.loc[:, column1] - df.loc[:, column2]).astype(float)
    return (column1 + '_minus_' + column2)


def make_ratio_columns(df, column1, column2):
    """Add a column that is computes the ratio column1 / column2.

    Usage:

    make_percent_between_columns(df,'sale_price','forecast_value').

    The new column in this example will be 'sale_price_div_forecast_value'.

    The function returns this new column name.
    """
    col_name = column1 + '_div_' + column2
    df.loc[:,
           col_name] = (df.loc[:, column1] / df.loc[:, column2]).astype(float)
    return (col_name)


class PredictBid:

    def __init__(self, db_name='oh_hell.db') -> None:
        con = create_engine(f"sqlite:///{db_name}")
        self.df_rounds = pd.read_sql("Select * from rounds_view; ", con=con)

        self.df_scores = pd.read_sql("Select * from scores_view; ", con=con)
        make_delta_columns(self.df_scores, 'taken', 'bid')
        make_delta_columns(self.df_scores, 'total_bid', 'total_cards')
        make_delta_columns(self.df_scores, 'bid', 'total_cards')
        make_ratio_columns(self.df_scores, 'bid', 'total_cards')
        self.player_map = pd.read_sql("select * from player_names", con=con)

        self.prep_data()

    def prep_data(self):
        score_data_training = self.df_scores.set_index(
            ['unique_hash', 'player', 'round'])[[
                'taken_minus_bid',
                'bid_div_total_cards',
                'total_cards',
                'total_bid_minus_total_cards',
            ]]
        double_deck_map = score_data_training.reset_index().groupby(
            ['unique_hash'])['total_cards'].max()
        player_count = score_data_training.reset_index().groupby(
            ['unique_hash'])['player'].nunique()

        deck_map = (double_deck_map * player_count) // 52

        trump_data = pd.get_dummies(
            self.df_rounds.query('is_trump == True')['card_rank'],
            prefix='trump').astype('int')

        singleton_data = pd.get_dummies(
            self.df_rounds.query('n_cards_suit_round == 1')['card_rank'],
            prefix='singles').astype('int')
        score_data_training['made_bid'] = (
            score_data_training['taken_minus_bid'] == 0).astype(int)
        df_training = pd.get_dummies(self.df_rounds, columns=['card_rank'])

        training_dfs = [
            pd.get_dummies(
                self.df_rounds.query("card_suit == @suit")['card_rank'],
                columns=['card_rank'],
                prefix=f"card_rank_{suit}").fillna(0).astype(int)
            for suit in ['hearts', 'clubs', 'diamonds', 'spades']
        ]
        df_training = pd.concat(training_dfs).fillna(0).astype(int)
        df_training = self.df_rounds.join(df_training)
        df_training = df_training.join(trump_data).fillna(0)
        df_training = df_training.join(singleton_data).fillna(0)
        print(df_training.columns)
        cols = ['is_trump'] + [
            col for col in df_training.columns if col.startswith('card_rank_')
        ] + list(trump_data.columns) + list(singleton_data.columns)
        df_training_cards = df_training.groupby(
            ['unique_hash', 'player', 'round'])[cols].sum()
        self.final_training = df_training_cards.join(
            score_data_training).reset_index()
        self.final_training['n_decks'] = self.final_training[
            'unique_hash'].map(deck_map)
        dealer_map = self.df_rounds.query('hand == 1').set_index(
            ['unique_hash', 'round', 'player'])['card_order']
        self.final_training['player_count'] = self.final_training[
            'unique_hash'].map(player_count)
        self.final_training = self.final_training.set_index([
            'unique_hash', 'round', 'player'
        ]).join(dealer_map).reset_index().query('total_cards >= 4')

    def _upload(self, data: pd.DataFrame, table_name: str, unique_hash: str):
        engine = create_engine("sqlite:///oh_hell.db")
        with engine.connect() as con:
            try:
                hashes = {
                    x[0]
                    for x in con.execute(
                        text(
                            f"select distinct (unique_hash) from {table_name};"
                        ))
                }
            except OperationalError:
                print('hash fail')
                hashes = set()

        if unique_hash in hashes:
            print(f"game {unique_hash }already in DB")
            return f'game already in {table_name} DB {unique_hash}'

        data.to_sql(table_name, con=engine, if_exists='append', index=False)
        return f'success for {table_name}'

    def train_and_upload_all(self):
        engine = create_engine("sqlite:///oh_hell.db")
        with engine.connect() as con:
            try:
                hashes = {
                    x[0]
                    for x in con.execute(
                        text(
                            f"select distinct (unique_hash) from predictions_detail;"
                        ))
                }
            except OperationalError:
                print('hash fail')
                hashes = set()
        score_hash_set = set(self.df_scores['unique_hash'].unique())
        for hash in tqdm(score_hash_set.difference(hashes)):
            self.train(hash)

    def make_all_alt_bids(self, all_data: pd.DataFrame, xgb, cols_train):
        return pd.concat([
            self.make_alt_bids(data, xgb, cols_train=cols_train)
            for _player, data in all_data.groupby(['player', 'round'])
        ])

    @staticmethod
    def make_alt_bids(one_player: pd.DataFrame, xgb, cols_train):
        one_player['bid'] = (one_player['bid_div_total_cards'] *
                             one_player['total_cards']).astype(int)
        rec = one_player.iloc[0]
        other_bids = rec['total_bid_minus_total_cards'] + rec[
            'total_cards'] - rec['bid']
        bid_range = pd.Series(range(0, rec['total_cards'] + 1))
        N = rec['total_cards'] + 1
        df = pd.concat(([one_player] * N)).reset_index(drop=True)
        df['bid'] = bid_range
        df['bid_div_total_cards'] = df['bid'] / df['total_cards']
        df['total_bid_minus_total_cards'] = other_bids - df[
            'total_cards'] + bid_range
        df['prediction'] = xgb.predict_proba(df[cols_train])[:, 1]
        return df.query('total_bid_minus_total_cards != 0')

    def train(self, unique_hash, upload=True):
        """Train excluding one game to not overfit"""
        train_data = self.final_training.query("unique_hash != @unique_hash")
        test_data = self.final_training.query("unique_hash == @unique_hash")
        xgb = XGBClassifier(n_estimators=200,
                            min_child_weight=1.7,
                            eval_metric='logloss',
                            early_stopping_rounds=10,
                            learning_rate=.05,
                            max_depth=5,
                            objective='binary:logistic',
                            random_state=42)

        X = train_data[COLS_TRAIN]
        y = train_data['made_bid']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.05)
        xgb.fit(X_train.drop(columns=['taken_minus_bid'], errors='ignore'),
                y_train,
                eval_set=[(X_test, y_test)])
        if upload and not test_data.empty:
            test_data['prediction'] = xgb.predict_proba(
                test_data[COLS_TRAIN])[:, 1]

            max_prediction = self.make_all_alt_bids(test_data,
                                                    xgb,
                                                    cols_train=COLS_TRAIN)
        res = ''
        if upload:
            res += self._upload(test_data, 'predictions_detail',
                                unique_hash) + '\n'
            res += self._upload(max_prediction, 'prediction_alt_bids_detail',
                                unique_hash)
            return res
        else:
            return xgb


class PlayerInfluence():

    def __init__(self, db_name='oh_hell.db') -> None:
        self.db_name = db_name

    def process_data(self):
        con = create_engine(f"sqlite:///{self.db_name}")
        df = pd.read_sql("Select * from scores_view; ", con=con)
        df['max_round'] = df.groupby(['unique_hash'])['round'].transform(max)
        df['made_bid'] = (df['taken'] == df['bid'])

        df['made_bid_fraction'] = df.groupby(['player', 'unique_hash'
                                              ])['made_bid'].transform('mean')

        final_scores = df.query('round == max_round')
        temp_data = pd.get_dummies(final_scores,
                                   columns=['player_name'],
                                   prefix='player_name')
        mylist = [x for x in temp_data.columns if x.startswith('player_name')]
        fitting_data = temp_data.groupby('unique_hash')[mylist].sum()
        temp_fit_data = final_scores.set_index('unique_hash').join(
            fitting_data)
        temp_fit_data = final_scores.set_index('unique_hash').join(
            fitting_data)
        temp_fit_data['max_score'] = temp_fit_data.groupby(
            'unique_hash')['points'].transform(max)
        temp_fit_data['norm_score'] = temp_fit_data['points'] / temp_fit_data[
            'max_score']
        temp_fit_data['win_flag'] = (
            temp_fit_data['points'] == temp_fit_data['max_score']).astype(int)
        temp_fit_data['score_rank'] = temp_fit_data.groupby(
            'unique_hash')['points'].rank(ascending=False)
        temp_fit_data['n_players'] = temp_fit_data.groupby(
            'unique_hash')['player'].transform('nunique')
        temp_fit_data.to_sql('linear_model_data', con=con, if_exists='replace')

    def get_linear_fit_menu_data(self,
                                 user,
                                 min_count=10,
                                 n=150,
                                 return_users=False):
        con = create_engine(f"sqlite:///{self.db_name}")
        temp_fit_data = pd.read_sql(
            "select * from linear_model_data where player_name = ? order by file_name DESC limit ?",
            params=(user, n),
            con=con)
        cols = [
            x for x in temp_fit_data.columns if x.startswith('player_name_')
        ]
        other_cols = [
            x for x in temp_fit_data.columns
            if not x.startswith('player_name_')
        ]
        col_series = (temp_fit_data[cols].sum() > min_count)

        qualified_user_variables = {
            x: y
            for x, y in temp_fit_data[cols].sum().to_dict().items()
            if y > min_count
        }

        if return_users:
            return qualified_user_variables
        return temp_fit_data

    def build_model(self, thedata, name, dependent, fitting_variables):
        assert dependent in [
            'made_bid_fraction',
            'score_rank',
            'norm_score',
            'win_flag',
        ]
        fitting_variables = [
            x for x in fitting_variables if x != f"player_name_{name}"
        ]
        X = thedata[fitting_variables]
        X = sm.add_constant(X)
        y = thedata[dependent]
        model = sm.OLS(y, X)
        results = model.fit()
        return results.summary2().tables[1].sort_values('Coef.')


if __name__ == '__main__':
    m = PlayerInfluence()
    m.process_data()
