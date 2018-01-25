# Install

Make sure pem/ is importable

# Data file

make sure models exist in this folder

data/data.tar.gz

# Create frames

See attached ipython notebooks for examples

# Create movie

I used the following command line tool to produce movies from my png frames.

ffmpeg -f image2 -i test-%04d.png -vcodec mpeg4 -r 12 -b 5000k test.mp4