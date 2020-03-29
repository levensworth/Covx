echo 'Installing virtualenv'
pip3 install virtualenv
echo 'creating a venv'
virtualenv -p python3.7 venv
echo 'activating venv'
. ./venv/bin/activate
echo 'installing requirements.txt'
pip3 install -r requirements.txt
echo 'start server ...'
python3 app/app.py
