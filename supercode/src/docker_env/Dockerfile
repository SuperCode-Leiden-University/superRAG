FROM debian:13
# Alpine is a lightweight Linux distribution

RUN apt-get update 

# install basic packages

# order: sudo/utilities, pip, C/C++ compilers, Python
RUN apt-get install -y time libffi-dev libssl-dev build-essential \
                       python3 python3-pip python3-dev python3.13-venv
#RUN apt-get install -y nodejs npm # Node.js
#RUN apt-get install -y openjdk-17-jdk # Java
#RUN apt-get install -y curl && curl https://sh.rustup.rs -sSf | sh -s -- -y # Rust

# create a virtual environment
RUN python3 -m venv /opt/venv

# ensure venv Python and pip are used
ENV PATH="/opt/venv/bin:$PATH"

# install HumanEval evaluator inside the venv
RUN pip install --no-cache-dir datasets human-eval

# copy all the files inside results to the container
WORKDIR /workspace

# add and set my agent as a user
#RUN useradd my_agent
#USER my_agent

# to run the application, IMPORTANT: not even comments after this command!!!
#CMD ["python3", "results/compile_gen_code.py"]
