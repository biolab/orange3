# Install build requirements

python -m ensurepip
python -m pip install pip==8.1.* wheel==0.29.*

python -m pip install `
    --extra-index-url $env:STAGING_INDEX `
    --only-binary numpy `
    numpy==$env:NUMPY_BUILD_VERSION

if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }
