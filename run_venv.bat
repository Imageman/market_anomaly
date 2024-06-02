
:: Batch script to setup a virtual development environment

@echo off

:: Variables
set venv_folder=.\.venv
set script_name=main.py 


if exist %venv_folder%\ (
    echo Virtual environment already exists
    echo.
    goto activate
) else (
    :: Creating virtual environment
    echo Creating virtual environment
    echo.

    python -m venv "%venv_folder%"
    call %venv_folder%\Scripts\activate.bat
    :: Update pip
    echo Updating pip in virtual environment
    echo.
    python.exe -m pip install --upgrade pip

    echo Install requirements.txt in virtual environment
    %venv_folder%\Scripts\pip install -r requirements.txt
)

:: Activating virtual environment
:activate
echo Activating virtual environment
echo.

call %venv_folder%\Scripts\activate.bat

:end
.\.venv\Scripts\python.exe "%script_name%"

if exist result.txt (
  set /p result=<result.txt
  c:\Users\Vovik\Dropbox\Appl\Telega\postHour.bat %result%
)

del d:\Vovik\Docs\Public\finance_plot.html 
copy plot.html d:\Vovik\Docs\Public\finance_plot.html

d:
cd d:\Vovik\Docs\Public\
@Echo Добавляем изменения в локальный git
git add --all
git commit -m "Automatic add finance_plot.html"
@Echo Отсылаем изменения на github
git push

timeout 5