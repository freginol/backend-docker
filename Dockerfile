

FROM python:3.8
WORKDIR '/app'
COPY . .
RUN pip install flask
RUN pip install pickle-mixin
RUN pip install pandas
RUN pip install numpy
RUN pip install scipy
RUN pip install lime
ENTRYPOINT ["python"]
CMD ["flaskapp.py"]

