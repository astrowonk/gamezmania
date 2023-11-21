CREATE INDEX IF NOT EXISTS file_name_scores on scores(file_name);

CREATE INDEX IF NOT EXISTS file_name_rounds on rounds(file_name);

CREATE INDEX IF NOT EXISTS unique_hashrounds on rounds(unique_hash);

CREATE INDEX IF NOT EXISTS unique_hashscores on scores(unique_hash);

CREATE INDEX IF NOT EXISTS pdpu on predictions_detail(player, unique_hash);

CREATE INDEX IF NOT EXISTS pred_hash on predictions_detail(unique_hash);

CREATE INDEX IF NOT EXISTS pred_alt_bid on prediction_alt_bids_detail(unique_hash);

CREATE INDEX IF NOT EXISTS player_hash_scores on scores(unique_hash, player);

CREATE INDEX IF NOT EXISTS player_scores on scores(player);