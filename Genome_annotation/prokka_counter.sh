count=$(find . -type d ! -path . | wc -l)
echo "The total number of prokka annotation folders is $count"
