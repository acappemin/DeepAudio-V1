

#test_lst=./eval/v2c_test.lst
#outputs=./eval/outputs_v2c_l44/

test_lst=./tests/v2c_test.lst
outputs=./tests/outputs_v2c_l44_test/


python ./eval/spk_wav2vec.py ${test_lst} ${outputs}

python ./eval/emo_wav2vec.py ${test_lst} ${outputs}

python ./eval/mcd_test.py ${test_lst} ${outputs} dtw

python ./eval/mcd_test.py ${test_lst} ${outputs} dtw_sl

python ./F5-TTS/src/f5_tts/eval/eval_v2c_test.py --eval_task wer --gen_wav_dir ${outputs}/gen/ --librispeech_test_clean_path ${outputs}/ref/ --metalst ${test_lst} --gpu_nums 8

#python ./F5-TTS/src/f5_tts/eval/eval_v2c_test.py --eval_task sim --gen_wav_dir ${outputs}/gen/ --librispeech_test_clean_path ${outputs}/ref/ --metalst ${test_lst} --gpu_nums 8

