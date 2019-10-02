# Tensorflow Demo Machine Learning Flask App

## Description
This is a machine learning modeul used to determine the sentiment of a movie review. It is based off of this project https://github.com/groveco/content-engine. Included is a CSV of IMDB review data which can be used to train the model. The application has been setup with Docker for ease of development and deployment.

### To run the application locally:

1. Install docker with docker compose on your machine
2. Clone this repo
3. From within the project root run:
> docker-compose up -d

4. Once that is complete verify the flask container and nginx containers have started:
> docker ps

5. You should also be able to see the application running at http://localhost:5000/
6. Next connect to the container and train the model:
> docker exec -it tensorflowdemo_flask_1 python3 modeltrain.py

This will train the model and in the end output a training set accuracy and test set accuracy.

7. Now you can start submitting data to the model:
> curl -X POST -H "X-API-TOKEN: FOOBAR1" -H "Content-Type: application/json; charset=utf-8" http://localhost:5000/tensorflowpredict -d '{"review": "No one expects the Star Trek movies to be high art, but the fans do expect a movie that is as good as some of the best episodes. Unfortunately, this movie had a muddled, implausible plot that just left me cringing - this is by far the worst of the nine (so far) movies. Even the chance to watch the well known characters interact in another movie cant save this movie including the goofy scenes with Kirk, Spock and McCoy at Yosemite. I would say this movie is not worth a rental, and hardly worth watching, however for the True Fan who needs to see all the movies, renting this movie is about the only way youll see it even the cable channels avoid this movie."}'

The prediction returned will either be 1 or 0. 1 indicates a positive movie review and 0 indicates a negative movie review.

### Docker Notes

#### List running containers
> docker ps

#### Rebuild containers from scratch
> docker-compose build --no-cache

### Other Helpful Commands

#### Kill the uwsgi server
> docker exec -it tensorflowdemo_flask_1 fuser -k 5000/tcp
> docker-compose up -d

https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub
