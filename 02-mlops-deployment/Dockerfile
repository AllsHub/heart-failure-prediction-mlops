FROM tensorflow/serving:latest

COPY ./serving_model /models/heart_failure_model
COPY ./tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh
COPY ./prometheus.config /models/prometheus.config

RUN sed -i 's/\r$//' /usr/bin/tf_serving_entrypoint.sh
RUN chmod +x /usr/bin/tf_serving_entrypoint.sh

ENV MODEL_NAME=heart_failure_model
ENV MODEL_BASE_PATH=/models

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]