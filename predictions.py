import pandas as pd
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sqlalchemy.exc import OperationalError
from tqdm.notebook import tqdm


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

    def __init__(self) -> None:
        con = create_engine("sqlite:///oh_hell.db")
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
        trump_data = pd.get_dummies(
            self.df_rounds.query('is_trump == True')['card_rank'],
            prefix='trump').astype('int')
        score_data_training['made_bid'] = (
            score_data_training['taken_minus_bid'] == 0).astype(int)
        df_training = pd.get_dummies(self.df_rounds, columns=['card_rank'])
        df_training = df_training.join(trump_data).fillna(0)
        cols = ['is_trump'] + [
            col for col in df_training.columns if col.startswith('card_rank_')
        ] + list(trump_data.columns)
        df_training_cards = df_training.groupby(
            ['unique_hash', 'player', 'round'])[cols].sum()
        self.final_training = df_training_cards.join(
            score_data_training).reset_index()
        dealer_map = self.df_rounds.query('hand == 1').set_index(
            ['unique_hash', 'round', 'player'])['card_order']
        self.final_training = self.final_training.set_index(
            ['unique_hash', 'round', 'player']).join(dealer_map).reset_index()

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
        for hash in tqdm(self.df_scores['unique_hash'].unique()):
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
        print(other_bids)
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
        xgb = XGBClassifier(n_estimators=130,
                            min_child_weight=1.5,
                            eval_metric='logloss',
                            early_stopping_rounds=10,
                            learning_rate=.05,
                            max_depth=5,
                            objective='binary:logistic',
                            random_state=42)

        cols_train = [
            'is_trump',
            'card_rank_2',
            'card_rank_3',
            'card_rank_4',
            'card_rank_5',
            'card_rank_6',
            'card_rank_7',
            'card_rank_8',
            'card_rank_9',
            'card_rank_T',
            'card_rank_ace',
            'card_rank_jack',
            'card_rank_king',
            'card_rank_queen',
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
        ]
        X = train_data[cols_train]
        y = train_data['made_bid']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.05)
        xgb.fit(X_train.drop(columns=['taken_minus_bid'], errors='ignore'),
                y_train,
                eval_set=[(X_test, y_test)])
        test_data['prediction'] = xgb.predict_proba(test_data[cols_train])[:,
                                                                           1]

        max_prediction = self.make_all_alt_bids(test_data,
                                                xgb,
                                                cols_train=cols_train)
        res = ''
        if upload:
            res += self._upload(test_data, 'predictions_detail',
                                unique_hash) + '\n'
            res += self._upload(max_prediction, 'prediction_alt_bids_detail',
                                unique_hash)
            return res
        else:
            return xgb
