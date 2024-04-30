python -m zoning.data_processing.eval --num-eval-rows 30 --terms min_unit_size --search-method experiment_3 --extraction-method answer_confirm --k 10
python -m zoning.data_processing.eval --num-eval-rows 30 --terms min_lot_size --search-method experiment_3 --extraction-method answer_confirm --k 10

# Docker
docker build -t echocool/zoning-gpt .
docker push echocool/zoning-gpt

# docker-compose
docker-compose up -d

python -m zoning.data_processing.eval --num-eval-rows 30 --terms min_parking_spaces --search-method experiment_3 --extraction-method answer_confirm --k 10

# Coverage
python -m zoning.data_processing.eval --num-eval-rows 30 --terms max_lot_coverage --search-method experiment_3 --extraction-method answer_confirm --k 10


python -m zoning.data_processing.eval --num-eval-rows 30 --terms min_lot_size,min_unit_size,max_height,min_parking_spaces,max_lot_coverage --search-method experiment_3 --extraction-method answer_confirm --k 10

python -m zoning.data_processing.eval --num-eval-rows 30 --terms "floor_to_area_ratio" --search-method experiment_3 --extraction-method answer_confirm --k 10

python zoning/main/search.py | python zoning/main/extract.py | python zoning/main/table_printer.py
python zoning/main/search.py | python zoning/main/extract.py | python zoning/main/json_printer.py


python zoning/main/search.py --term "min lot size" --town "andover" --district_name "Andover Lake" --district_abb "AL" --search_method "experiment_3" --k 10 | python zoning/main/extract.py --extraction_method "answer_confirm" --model_name "gpt-4-1106-preview" --tournament_k 10 | python zoning/main/table_printer.py

pdm run python -m zoning.main.search | pdm run python -m zoning.main.extract | pdm run python -m zoning.main.table_printer

pdm run python -m zoning.main.search --term "min lot size" --town "andover" --district_name "Andover Lake" --district_abb "AL" --search_method "experiment_3" --k 10