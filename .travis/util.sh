
foldable ()
{
    # As documented in:
    # https://github.com/travis-ci/travis-ci/issues/2158#issuecomment-42726890
    # https://github.com/travis-ci/travis-ci/issues/2285#issuecomment-42724719
    local _id="$RANDOM$RANDOM$RANDOM"
    echo "travis_fold:start:$_id"
    echo "$*"
    "$@"
    local _estatus="$?"
    echo "travis_fold:end:$_id"
    return $_estatus
}
