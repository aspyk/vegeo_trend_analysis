FROM python:3.10-slim-bullseye

COPY QMapp/requirements.txt /tmp/
RUN python -m pip install -r /tmp/requirements.txt

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY QMapp/main.py QMapp/main.py
COPY c3s_land_mask.h5 QMapp/data/c3s_land_mask.h5

CMD ["bokeh", "serve", "QMapp"]

