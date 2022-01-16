coverage run -m pytest
coverage-badge -o doc/coverage.svg
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