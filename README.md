# Linux/macOS
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1


# install lib
pip install -r requirements.txt

# deactive venv
deactivate

# build image 
docker build -t ocop-recommend-app .

# login dockerhub
docker login

# tag docker image
docker tag ocop-recommend-app ${your_dockerhub_username}/ocop-recommend-app:latest

# push docker hub
docker push ${your_dockerhub_username}/ocop-recommend-app:latest

