
echo "Setting up a testing env in $env:USERPROFILE\testenv"

python -m venv --clear $env:USERPROFILE\testenv

& $env:USERPROFILE\testenv\Scripts\Activate.ps1

$version = python setup.py --version

python -c 'import sys; print(\"sys.prefix:\", sys.prefix); print(sys.version)'
python -m pip install pip==8.1.* wheel==0.29.*

# run install and test from a empty dir to avoid imports from current dir

mkdir -force build/testdir | out-null
pushd

try {
    cd build/testdir

    # Install numpy/scipy from staging index (contains numpy and scipy
    # extracted form the legacy superpack installers (sse2 builds))

    python -m pip install `
        --index-url "$env:STAGING_INDEX" `
        --only-binary "numpy,scipy" numpy scipy

    if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }


    # Install specific Orange3 version
    python -m pip install --no-deps --no-index `
        --find-links ../../dist `
        Orange3==$version

    if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

    # Instal other remaining dependencies
    python -m pip install `
        --extra-index-url "$env:STAGING_INDEX" `
        --only-binary "numpy,scipy" `
        Orange3==$version

    if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

    echo "Test environment:"
    echo "-----------------"
    python -m pip freeze
    echo "-----------------"

    # Run core tests
    echo "Running tests"
    echo "-------------"
    python -m unittest -v Orange.tests

    if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

    # Widget tests
    python -m pip install `
        --index-url "$Env:STAGING_INDEX" `
        PyQt5

    echo "Running widget tests with PyQt5"
    echo "-------------------------------"
    try {
        $Env:ANYQT_HOOK_BACKPORT = "pyqt4"
        python -m unittest -v Orange.widgets.tests
        if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }
    } finally {
        $Env:ANYQT_HOOK_BACKPORT = ""
    }

    python -m pip uninstall --yes PyQt5
    python -m pip install `
        --extra-index-url "$Env:STAGING_INDEX" `
        PyQt4

    echo "Running widget tests with PyQt4"
    echo "-------------------------------"

    python -m unittest -v Orange.widgets.tests
    if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

} finally {
    popd
}
