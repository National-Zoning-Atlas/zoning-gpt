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
