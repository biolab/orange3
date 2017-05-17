
echo "Building and testing using conda in: $env:MINICONDA"
echo ""

if (-not($env:MINICONDA)) { throw "MINICONDA env variable must be defined" }

$python = "$env:MINICONDA\python"
$conda = "$env:MINICONDA\Scripts\conda"

# Need at least conda 4.1.0 (channel priorities)

& "$conda" install --yes "conda>=4.1.0"

# add conda-forge channel
& "$conda" config --append channels conda-forge

# some required packages that are not on conda-forge
& "$conda" config --append channels ales-erjavec

& "$conda" install --yes conda-build

echo "Conda info"
echo "----------"
& "$conda" info
echo ""

echo "Starting conda build"
echo "--------------------"
& "$conda" build conda-recipe

if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

# also copy build conda pacakge to build artifacts
echo "copying conda package to dist/conda"
echo "-----------------------------------"

$pkgpath = & "$conda" build --output conda-recipe
mkdir -force dist/conda | out-null
cp "$pkgpath" dist/conda/
