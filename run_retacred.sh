for SEED in 4 11 40 42 67;
do
  python train_retacred.py --model_name_or_path roberta-large --data_dir ./dataset/retacred --output_dir saved_models/da-gdpn-retacred-$SEED --input_format typed_entity_marker_punct --seed $SEED --run_name retacred;
done;