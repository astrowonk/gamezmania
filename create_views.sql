
CREATE VIEW rounds_view as
select
	r.*,
	case
		when p.player_name is null then r.player
		else p.player_name
	end as player_name
from
	rounds r
left join player_names p on
	r.player = p.player_id
/* rounds_view(round,player,card_rank,card_suit,card_order,hand,winner,tram,bad_card,is_trump,n_cards_suit_round,bad_singleton,unique_hash,file_name,ace_flag,player_name) */;


CREATE VIEW scores_view as select r.*, case
		when p.player_name is null then r.player
		else p.player_name
	end as player_name from scores r left join player_names p on r.player = p.player_id
/* scores_view(round,player,points,round_points,file_name,unique_hash,bid,taken,total_cards,total_bid,player_name) */;
CREATE VIEW bids_view as select r.*, case
		when p.player_name is null then r.player
		else p.player_name
	end as player_name  from bids r left join player_names p on r.player = p.player_id
/* bids_view(round,player,bid,bad_singleton,unique_hash,file_name,player_name) */;
CREATE VIEW last_played as select player, max(file_name) as last_played_game from scores_view sv group by player
/* last_played(player,last_played_game) */;
CREATE VIEW lucky_or_good as select * from (select
	svv.player,svv.player_name,last_played_game,
	avg(made_bid-prediction) as made_bid_minus_prediction, count(made_bid) as game_count
from
	(
	select
		player,
		player_name,
		unique_hash,
		sum(case when bid = taken then 1 else 0 end) as made_bid
from
		scores_view sv
group by
		player,
		unique_hash) svv
left join predictions p on
	p.unique_hash = svv.unique_hash
	and p.player == svv.player left join last_played lp on svv.player = lp.player group by svv.player,svv.player_name) where game_count > 10 order by 4
/* lucky_or_good(player,player_name,last_played_game,made_bid_minus_prediction,game_count) */;
CREATE VIEW predictions as select player, unique_hash, sum(prediction) as prediction from predictions_detail group by player, unique_hash
/* predictions(player,unique_hash,prediction) */;

CREATE VIEW predictions_detail_view as select
        case
          when pn.player_name is null then pd.player
          else pn.player_name
        end as player_name ,
        unique_hash,
        round,
       cast (card_order as integer) card_order,
        total_bid_minus_total_cards,
        cast ((total_cards * bid_div_total_cards) as integer) as bid,
        total_cards,
        made_bid,taken_minus_bid,
        prediction,is_trump
from
        predictions_detail pd left join player_names pn on pn.player_id = pd.player
/* predictions_detail_view(player_name,unique_hash,round,card_order,total_bid_minus_total_cards,bid,total_cards,made_bid,taken_minus_bid,prediction,is_trump) */;
