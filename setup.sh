git clone https://github.com/NVlabs/stylegan3.git
mv stylegan3/* .
mkdir models
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl
mv stylegan3-r-ffhq-1024x1024.pkl models
wget wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl
mv vgg16.pkl models
