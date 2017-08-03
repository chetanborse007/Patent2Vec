mkdir ~/python
cd ~/python
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar xvzf Python-3.5.2.tgz
cd Python-3.5.2/
./configure --prefix=$HOME/python/
make && make install

# Edit ~/.bash_profile as below,
#PATH=$PATH:$HOME/bin
#PATH=$HOME/python/Python-3.5.2/:$HOME/.local/bin:$PATH
#PYTHONPATH=$HOME/python/Python-3.5.2
#export PATH
#export PYTHONPATH

source ~/.bash_profile

wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py -O - | python - --user

pip install --user --upgrade gensim

pip install --user --upgrade nltk
