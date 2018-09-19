#stop container
docker stop $USER-beaker-face-of-crystals-2017
# remove container
docker rm $USER-beaker-face-of-crystals-2017

# pull the image from dockerhub
docker pull ziletti/face-of-crystals-2017:v2.0.1

# run it in labdev. Be sure to mount the data in the appropriate locations.
#Labdev
#docker run --restart=unless-stopped -d -p 9009:8801 --name=$USER-beaker-face-of-crystals-2017 ziletti/face-of-crystals-2017:v2.0.1
#nomadteam
#docker run --restart=unless-stopped -d -p 9009:8801 --name=ziang-beaker-face-of-crystals-2017 -v /nomad-fast/nomadlab/parsed/prod-028/VaspRunParser1.3.0-7-g615671f/:/parsed/production/VaspRunParser1.2.0-3-g4facbeb:ro  ziletti/face-of-crystals-2017:v2.0.1
# local
docker run --restart=unless-stopped -d -p 8801:8801 --name=$USER-beaker-face-of-crystals-2017 ziletti/face-of-crystals-2017:v2.0.1


#docker run --restart=unless-stopped -d -p 9009:8801 --name=$USER-beaker-face-of-crystals face-of-crystals:v1.0.0
