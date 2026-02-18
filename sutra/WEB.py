import sys
from streamlit.web import cli as stcli
from sutra import   app_v7

def main():
    sys.argv = ["streamlit", "run", app_v7.__file__]
    sys.exit(stcli.main())

# if __name__ == "__main__":
#     main()