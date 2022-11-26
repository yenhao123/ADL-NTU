if [ ! -f glove.840B.300d.txt ]; then
  wget https://www.dropbox.com/s/5mavkgt0f0oeoia/glove.840B.300d.zip?dl=1 -O glove.840B.300d.zip
  unzip glove.840B.300d.zip
fi
python preprocess_intent.py
python preprocess_slot.py
