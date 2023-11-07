import pandas as pd
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sqlalchemy.exc import OperationalError


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
        score_data_training['made_bid'] = (
            score_data_training['taken_minus_bid'] == 0).astype(int)
        df_training = pd.get_dummies(self.df_rounds, columns=['card_rank'])
        cols = ['is_trump'] + [
            col for col in df_training.columns if col.startswith('card_rank_')
        ]
        df_training_cards = df_training.groupby(
            ['unique_hash', 'player', 'round'])[cols].sum()
        self.final_training = df_training_cards.join(
            score_data_training).reset_index()
        dealer_map = self.df_rounds.query('hand == 1').set_index(
            ['unique_hash', 'round', 'player'])['card_order']
        self.final_training = self.final_training.set_index(
            ['unique_hash', 'round', 'player']).join(dealer_map).reset_index()

    def _upload(self, data: pd.DataFrame, table_name: str, unique_hash: str):
        con = create_engine("sqlite:///oh_hell.db")
        try:
            hashes = [
                x[0] for x in con.execute(
                    f"select distinct (unique_hash) from {table_name};")
            ]
        except OperationalError:
            print('hash fail')
            hashes = set()

        if unique_hash in hashes:
            print(f"game {unique_hash }already in DB")
            return f'game already in {table_name} DB {unique_hash}'

        data.to_sql(table_name, con=con, if_exists='append', index=False)
        return f'success for {table_name}'

    def train(self, unique_hash):
        """Train excluding one game to not overfit"""
        train_data = self.final_training.query("unique_hash != @unique_hash")
        test_data = self.final_training.query("unique_hash == @unique_hash")
        xgb = XGBClassifier(n_estimators=100,
                            min_child_weight=1.2,
                            eval_metric='logloss',
                            early_stopping_rounds=10,
                            learning_rate=.15,
                            max_depth=3,
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
        con = create_engine("sqlite:///oh_hell.db")
        return self._upload(test_data, 'predictions_detail', unique_hash)
