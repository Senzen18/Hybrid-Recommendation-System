FROM python:3.11.9

COPY requirements.txt .

RUN pip3 install torch  --index-url https://download.pytorch.org/whl/cu121
RUN pip install -U scikit-learn
RUN pip install -r requirements.txt


ADD recommend.py .
ADD ttest.py .
ADD Recommender_Responsev1.csv .
ADD Recommender_Responsev2.csv .
COPY ./cleaned_100k ./cleaned_100k
COPY ./NCF_model ./NCF_model
ADD movies.npy .
ADD users.npy .
CMD ["python","./recommend.py"]
