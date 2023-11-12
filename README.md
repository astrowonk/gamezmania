
# Parsing Cardzmania json files

This started as a class that will load a json file from [cardzmania](https://www.cardzmania) XHR request and make a score line graph. This is a JSON data structure retrieved from a somewhat hidden API when looking at a historical game that can be found with the Inspect tool. (For example on the Profile page

## Oh Hell data parsing

### Creating useful dataframes and sending to a database:

The `parse_oh_data` and `make_score_dataframe` methods transform the raw JSON into round by round, sometimes hand by hand data. The `upload_to_sql` method uses both of these methods to make dataframes that are sent to 3 different sqlite tables for `bids`, `scores`,and `rounds` : the latter having the full record of each card played in each round.

### Views

Several useful views can be made from these underlying tables. I somewhat manually create a `player_names` table that maps the alphanumeric player id in the JSON to a real human name. This generally has to be done by inspection looking at the game on Cardzmania and then figuring out how the player order maps onto the list of player names in the top of the file. Eventually I will add a `create_views.sql` that creates all the views I use.

### Training a classifier model

Once sufficient data is in the `scores`, `bids`, and `rounds` tables the `PredictBid` class in `predictions.py` creates an Xgboost classifier model training on the likelihood of a bid being made given the cards in the hand, the bid, the dealer order, and how over or under bid the round is. These results get pushed to a `predictions_detail` table, where predictions for a given game are stored (trained on everything except that particular game.)





