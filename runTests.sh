coverage run -m pytest -v --db monitoring.db
coverage-badge -o doc/coverage.svg -f
coverage html 
echo "Open Report [Y/n]" 
read input
if [[($input == "" || $input == "Y" || $input == "y")]] 
then
    _currentDir="$(pwd)"
    _pathToOpen="file://$_currentDir/htmlcov/index.html"
    echo "Opening: $_pathToOpen" 
    python -mwebbrowser $_pathToOpen
fi