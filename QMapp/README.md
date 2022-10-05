# QMapp Bokeh app


## Docker based deployment


### Build

Build Docker image. 
Run something like the following from project root
(necessary because the `c3s_land_mask.h5` resource is at the project root):

    docker build -f QMapp/Dockerfile .


### Run

Run the Docker image (e.g. assuming `3f80b17c2b67`),
for example like this:

    docker run --rm -p5006:5006 3f80b17c2b67


Visit http://localhost:5006/QMapp
