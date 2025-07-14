FROM mageai/mageai:latest

WORKDIR /home/src

# Copy the entire project directory
COPY classical-composer-prediction/ ./classical-composer-prediction/
COPY requirements.txt .
COPY .env .

# Install dependencies
RUN pip install -r requirements.txt

EXPOSE 6789

# Set the project name
ENV MAGE_PROJECT_NAME=classical-composer-prediction

CMD ["sh", "-c", "mage start $MAGE_PROJECT_NAME"]