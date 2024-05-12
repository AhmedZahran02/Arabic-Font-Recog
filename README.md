# Arabic Font Recog
 
1. install the following dependencies:
    pip install opencv-contrib-python
    pip install xgboost
    pip install scikit-learn
    pip install matplotlib
    pip install Flask
    pip install requests
    pip install bz2file
    pip install scipy
    pip install pickle5
    pip install glob2

2. run the file "main.ipynb" as follows:
    2.1 set the values of hyperparameters 
    2.2 import libraries
    2.3 load model
    2.4 run the classifier and the feature extractor
    2.5 pedict labels
    2.6 observe accuracy

3. testing with the deployed server
    we already have deployed a testing server on https://aozahran2025.pythonanywhere.com/ , you can just go and
    use the simple UI we created to test images.

4. testing with local server:
    after performing items in section 2 and generating the models files, head to /server and set the models paths
    then run server_test.py using "python server_test.py", this will run a local flash server similar to the one we deployed.