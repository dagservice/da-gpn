for SEED in 4 11 40 42 67;
do
  python run_classifier.py  --seed $SEED --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file ./bert_base/vocab.txt   --bert_config_file ./bert_base/bert_config.json   --init_checkpoint ./bert_base/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1_$SEED  --gradient_accumulation_steps 6
  python evaluate.py --f1dev berts_f1_$SEED/logits_dev.txt --f1test berts_f1_$SEED/logits_test.txt
done;
