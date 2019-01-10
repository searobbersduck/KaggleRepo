python run_classifier.py --task_name=quora --do_train=true --do_eval=true --data_dir=./data1/ --vocab_file=./model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=./model/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./model/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=64 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/sent_output2/ --num_train_epochs 6