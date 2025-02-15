# diabetes_predictor_machine_learning
Machine Learning Diabetes Predictor with web interface.


# INSTALL

Open a Terminal in Linux, you need to have python and pyenv installed:

        How to install pyenv:

                    https://realpython.com/intro-to-pyenv/
                    https://bbs.archlinux.org/viewtopic.php?id=262833

After that, execute these commands in your terminal:


    PYTHON_VER="3.12.8"

    pyenv install --list

    pyenv install $PYTHON_VER

    pyenv local $PYTHON_VER

    python -m venv --upgrade-deps venv

    source venv/bin/activate

    pip install -r requirements.txt

    ./start.sh


Open a web browser pointing to:

        http://localhost:5000


And follow instructions.
