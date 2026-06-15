import sys
from streamlit.web import cli as stcli
from sutra import SutraWEB

def main():
    sys.argv = ["streamlit", "run", SutraWEB.__file__]
    sys.exit(stcli.main())

# if __name__ == "__main__":
#     main()