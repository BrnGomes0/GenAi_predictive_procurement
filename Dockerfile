FROM python:3.13

WORKDIR /app

COPY . /app

RUN pip install uv && uv pip install -r requirements.txt --system

EXPOSE 5000

CMD ["python", "run.py"]