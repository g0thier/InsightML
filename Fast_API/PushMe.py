import os 
import time 
from datetime import datetime
import re
'''
print('')
checkInstall = input("Do you want I install heroku ?").lower()
print('')

if checkInstall == "yes" or checkInstall == 'y':
    os.system("brew tap heroku/brew && brew install heroku")

print('')
now = str( datetime.today().strftime("%y%m%d%H%M%S") )
name_input = input("A name for heroku app (only letters):").lower()
name = f'{name_input}{now}'
print('')

os.system('docker')
time.sleep(2)
os.system(f'docker build . -t applicationdocker{now}')
# docker run -d --name mycontainer -p 80:80 myimage # On Original start run.sh

os.system('heroku login')
time.sleep(10)
os.system("heroku container:login")

os.system(f"heroku create {name}")
os.system(f"heroku container:push web -a {name}")
os.system(f"heroku container:release web -a {name}")
os.system(f"heroku open -a {name}")
'''

os.system('docker')
time.sleep(2)
os.system(f'docker build . -t insightml')
os.system(f'docker run -p 8000:8000 insightml')

