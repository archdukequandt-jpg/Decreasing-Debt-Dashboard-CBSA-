\
    @echo off
    REM run_app.bat - Windows launcher for the Streamlit app
    REM Credits:
    REM   Creators: Ryan Childs (ryanchilds10@gmail.com) · James Quandt (archdukequandt@gmail.com) · James Belhund (jamesbelhund@gmail.com)

    cd /d "%~dp0"
    python -m streamlit run app.py
