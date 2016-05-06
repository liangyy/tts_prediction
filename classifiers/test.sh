echo "###########################################################"
echo Training a new classifier called 'test_classifier' for test
python tts_prediction_train.py training_datasets/test_training.txt training_datasets/test_test.txt motifs/ save test_classifier > tmp.out
if [ $? -eq 0 ]; then
    echo tts_prediction_train.py SUCCEEDS!!
else
    echo tts_prediction_train.py FAILS!!
fi
rm tmp.out
echo "##########################################################"
echo Predicting TTS on test_genome.fa
python tts_prediction.py test_classifier genome/test_genome.fa motifs/ 1 tmp.out
if [ $? -eq 0 ]; then
    echo tts_prediction.py SUCCEEDS!!
else
    echo tts_prediction.py FAILS!!
fi
rm tmp.out
echo "#########################################################"
